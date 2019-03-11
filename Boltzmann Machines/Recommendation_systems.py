#Importing the libraries
#We will be using the pytorch framework to build our Boltzman Machine
#based recommendation system that predicts user preference 
#and movie ratings.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Importing the dataset
movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None, engine='python',encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None, engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None, engine='python',encoding='latin-1')

#Preparing the training set and the test set
training_set=pd.read_csv('ml-100k/u1.base',delimiter='\t')
#Converting the training set into an array
training_set=np.array(training_set, dtype='int')

test_set=pd.read_csv('ml-100k/u1.test',delimiter='\t')
#Converting the test set into an array
test_set=np.array(test_set, dtype='int')

#Getting the number of users and movies
#Converting the training and test set to create a matrix
#Creating a matrix where the lines are the users and the columns are the movies 
#The cells are going to be ratings
nb_users= int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies= int(max(max(training_set[:,1]),max(test_set[:,1])))

def convert(data):
    #Creating a list of lists:943 lists
    new_data=[]
    for id_users in range(1,nb_users+1):
        id_movies=data[:,1][data[:,0] == id_users]
        id_ratings=data[:,2][data[:,0] == id_users]
        #Adding zero to those ratings that the user didn't rate
        ratings=np.zeros(nb_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append(list(ratings))
    return new_data

#List of list: Each list of 943 elements contains 1682 ratings of the movies
training_set=convert(training_set)
test_set=convert(test_set)

#Converting the list of lists into torch tensors
#Tensors are arrays that contains elements of a single datatype
#These multi-dimensional tensors are called pytorch tensors
training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)

#Converting all the ratings into its binarized format
# 1: Like 0:Dislike
#Restricted Boltzmann Machines
#Converting the ratings that were 0 in the original dataset to -1
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1


test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#Creating the architecture of the neural network
#Building a probabilistic graphical model

#Number of hidden nodes
#Weights and bias:Probability of hidden nodes given the visible nodes
#Weights and Bias:Probability of the visible nodes given the hidden nodes
#nv: number of visible nodes
#nh: number of hidden nodes
class RBM():
    def __init__(self,nv,nh):
        #initializing the parameters
        #variables of the object
        #randn: random normal distribution of mean=0 and variance=1
        self.W=torch.randn(nh,nv)
        self.bh=torch.randn(1,nh) #batch and bias
        self.bv=torch.randn(1,nv)
    #Sampling the hidden nodes based on the probabilities
    #Gibs Sampling- log likelihood gradients
    #Activation is based on the probabilities of hidden noden given the visible node
    #Probability here corresponds to sigmoid activation function
    #x: visible neurons-input vector of observations
    def sample_h(self,x):
        #Sigmois activation is applied to W * x + bias
        #torch.mm:Product of 2 torch tensors
        wx = torch.mm(x,self.W.t())
        activation=wx+self.bh.expand_as(wx)
        #bias is applied to each line of the mini batch
        p_h_given_v=torch.sigmoid(activation)
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    def sample_v(self,y):
        #Sample_v returns the probabilities of the visible node given the hidden node
        wy = torch.mm(y,self.W)
        activation=wy+self.bv.expand_as(wy)
        #bias is applied to each line of the mini batch
        p_v_given_h=torch.sigmoid(activation)
        return p_v_given_h,torch.bernoulli(p_v_given_h)
    def train(self,v0,vk,ph0,phk):
        #v0: input nodes
        #vk:visible node after k iterations
        #Contrastive divergence:approximating the log likelihood gradient
        #Energy function depends on the weights of the model
        #Goal: Maximize log likelihood, therefore compute gradient(approximate)
        #Gibbs sampling: sampling k times the hidden node and visible node
        
        self.W +=(torch.mm(v0.t(), ph0) - torch.mm(vk.t(),phk)).t()
        self.bv += torch.sum((v0-vk),0)
        self.bh += torch.sum((ph0-phk),0)
        
nv = len(training_set[0])
#detecting the features, nh is tunable
nh= 100
batch_size=100
#creating the RBM object
rbm = RBM(nv,nh)


#Training the RBM
nb_epoch = 10
for epoch in range(1,nb_epoch+1):
    #loss function to calculate the difference between actual and predicted values
    train_loss = 0
    count = 0.
    for id_user in range(0, nb_users - batch_size,batch_size):
        vk=training_set[id_user:id_user+batch_size]
        v0=training_set[id_user:id_user+batch_size] #target
        ph0,_=rbm.sample_h(v0)
        for k in range(10):
            #k-step contrastive divergence
            _,hk=rbm.sample_h(vk)
            _,vk=rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0] #keeping the -1 ratings
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        count += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/count))

# Testing the RBM
test_loss = 0
count = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        count += 1.
print('test loss: '+str(test_loss/count))
