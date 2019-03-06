#Predicting the rising and falling trends of Stock price in a market. 
#Dataset: Google Stock Price Data from 2012-2016

#Import the libraries including the Keras libraries and packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

#Loading the training data using Pandas
training_data=pd.read_csv('Google_Stock_Price_Train.csv')
train_set=training_data.iloc[:,1:2].values
plt.plot(train_set)
plt.title('Stock Price trend')
plt.xlabel('Time')
plt.ylabel('Opening price')

#Data preprocessing
#Normalizing the dataset
s=MinMaxScaler(feature_range=(0,1))
train_scale=s.fit_transform(train_set)

#Creating a data structure with 60 timesteps and 1 output
X_train=[]
y_train=[]
for i in range(60,len(train_set)):
    X_train.append(train_scale[i-60:i,0]) #A sliding window of size 60 with stride 1
    y_train.append(train_scale[i,0])
    
X_train,y_train=np.array(X_train),np.array(y_train)


#Reshaping the data into a acceptable tensor format
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

#Building the RNN
#Initializing the RNN

regressor=Sequential()

#Adding the LSTM layers into the model
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
# A certain percentage of neurons will be ignored at each iteration of training to avoid overfitting. 

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#Adding a fully connected layer into the network
regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the Regressor into the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#Predicting the values and Visualizing the data

#The Actual stock price values
test_data=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=test_data.iloc[:,1:2].values

#The predicted stock price values
dataset_total = pd.concat((training_data['Open'], test_data['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_data) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = s.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = s.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
