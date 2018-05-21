# IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import quandl
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import datetime
import math

# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)


#####
quandl.ApiConfig.api_key = 'API_KEY'

stock_name=input('enter stock (code) name:')
company_name = input('enter company (code) name:')

print("fetching data...")
data = quandl.get(stock_name+'/'+company_name, start_date='2011-01-01',
                  collapse='daily')

data.to_csv('a.csv', sep=',') 

#import data
data = pd.read_csv('a.csv')
print('data is ready.')
total_days = np.arange(1, len(data)+1,1)
#print(data[['Open','High','Low','Close']])



HLL_avg = data[['High', 'Low', 'Last']].mean(axis = 1)
#print(HLL_avg)
close_val = data[['Close']]

##plotting

plt.plot(total_days, HLL_avg, 'b', label = 'HLL avg')
plt.plot(total_days, close_val, 'g', label = 'Closing price')
plt.legend(loc = 'upper right')
plt.show()

## PREPARATION OF TIME SERIES DATASE
close_val = np.reshape(close_val.values, (len(close_val),1)) # 1664
scaler = MinMaxScaler(feature_range=(0, 1))
close_val = scaler.fit_transform(close_val)

# TRAIN-TEST SPLIT
train_close = int(len(close_val) * 0.75)
test_close = len(close_val) - train_close
train_close, test_close = close_val[0:train_close,:], close_val[train_close:len(close_val),:]

# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = new_dataset(train_close, 1)
#print(trainX)
testX, testY = new_dataset(test_close, 1)

# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1


# LSTM MODEL
model = Sequential()
model.add(LSTM(256, input_shape=(1, step_size), return_sequences = True))
model.add(LSTM(16))
model.add(Dense(1))
model.add(Activation('relu'))

# MODEL COMPILING AND TRAINING
model.compile(loss='mean_squared_error', optimizer='adagrad') 
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)


# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# TRAINING RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train RMSE: %.2f' % (trainScore))

# TEST RMSE
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE: %.2f' % (testScore))

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(close_val)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(close_val)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(step_size*2)+1:len(close_val)-1, :] = testPredict

# DE-NORMALIZING MAIN DATASET 
close_val = scaler.inverse_transform(close_val)

# PLOT OF MAIN  VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
plt.plot(close_val, 'g', label = 'original dataset')
plt.plot(trainPredictPlot, 'r', label = 'training set')
plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
plt.legend(loc = 'upper right')
plt.xlabel('Time in Days')
plt.ylabel('close value')
plt.show()

# PREDICT FUTURE VALUES
last_val = testPredict[-1]

next_val = model.predict(np.reshape(1, (1,1,1)))
print ("Last Day Value:", np.asscalar(last_val))
print ("Next Day Value:", np.asscalar(last_val*next_val))



