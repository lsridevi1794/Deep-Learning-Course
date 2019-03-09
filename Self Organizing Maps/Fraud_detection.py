# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

#Importing the MiniSom package
#MiniSom is a minimalistic numpy based implementation of Self_Organizing_Maps
from minisom import MiniSom

#Importing the sklearn library for feature scaling
from sklearn.preprocessing import MinMaxScaler

#Importing the dataset
#Import the dataset and split the labels from the other attributes and store it in variables X and y
dataset=pd.read_csv('Credit_Card_Applications.csv')
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

#Feature Scaling
s=MinMaxScaler(feature_range=(0,1))
X=s.fit_transform(X)

#Training the SOM 
#SOMs are 2 dimensional arrays that wrap over the entire dataset of high dimensionality
#SOMs detect the correlation between features 
som=MiniSom(x=15,y=15,input_len=15, sigma=1.0, learning_rate=0.5)
#Initializes the weights of the SOM picking random samples from data
som.random_weights_init(X)
# Trains the SOM picking samples at random from data 
som.train_random(data=X, num_iteration=150)

#Visualizing the SOM results
from pylab import bone, pcolor, colorbar, plot, show

bone()
# som.distance_map(): Returns the distance map of the weights.
# Each cell is the normalised sum of the distances between a neuron and its neighbours.
pcolor(som.distance_map())
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)                   # Computes the coordinates of the winning neuron for the sample x.
    #plotting the detected points in the center of the color spaces
    #w holds the co-ordinates of the edge, we are trying to place the detected points to the center in the colorbar
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],#The label values stored in y holds values either 0 or 1 indicating the approval of credit cards or not
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#From the map it is seen that we have one region marked white, this indicates a high mean inter neuron distance. 
# Finding the frauds
mappings = som.win_map(X)
#            Returns a dictionary wm where wm[(i,j)] is a list with all the patterns
#            that have been mapped in the position i,j.
frauds = np.concatenate((mappings[(11,1)],mappings[(13,8)]),axis=0)
frauds = s.inverse_transform(frauds)

#Further, we move on to finding the probabilities of fraud detection of customers 
#Since this is a hybrid model, we make use of the results obtained in the unsupervised model to train and generate our inferences
# Creating the matrix of features
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 5)

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]
