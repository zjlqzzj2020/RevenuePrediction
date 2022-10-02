#!/usr/bin/env python
# coding: utf-8

# ## Product Selection

# In[1]:
sheet = 'C1'
# sheet='C2'
# sheet='BYX'
# sheet='BZX'
# sheet='EDGA'
# sheet='EDGX_Equities'
# sheet='EXO_EDGX Options'
# sheet='OPT_BZX BATS'
# sheet='CFE'

#C1 Prediction 1m

#load ensemble
import os
from keras.models import load_model
from os import path






# ## 1m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '1','model_corr.h5'))


df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)

values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())

# split into train and test sets
values = reframed.values
n_train_months = 27
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble, model


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome1 = pd.DataFrame(outcome)
print(outcome1)

# In[3]:




# ## 2m

# In[4]:


from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '2','model_corr.h5'))

# sheet='BYX'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 27
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome2 = pd.DataFrame(outcome)
print('C2_2m')
print(outcome2)


# ## 3m

# In[5]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '3','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 27
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome3 = pd.DataFrame(outcome)
print('outcome3')
print(outcome3)


# ## 4m

# In[6]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '4','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
# #
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 27
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome4 = pd.DataFrame(outcome)
print(outcome4)


# ## 5m

# In[7]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '5','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 5
# split into train and test sets
values = reframed.values
n_train_months = 27
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome5 = pd.DataFrame(outcome)
print(outcome5)



# ## 6m

# In[12]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'
ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '6','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 27
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome6 = pd.DataFrame(outcome)
print(outcome6)


# ## 12m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '12','model_corr.h5'))

# sheet='OPT_BZX BATS'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
# n_months = 11
n_months = 23
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 9
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1) // 2) * n_features  # the following 12
# n_obs = (n_months+1)* n_features #the following
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1) // 2, n_features))
# train_X = train_X.reshape((train_X.shape[0], (n_months+1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1) // 2, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome12 = pd.DataFrame(outcome)
print(outcome12)


# ## Prediciton Chart

# In[14]:


# Concatenate all dataframes by identical columns

df = pd.concat([outcome1, outcome2, outcome3, outcome4, outcome5, outcome6, outcome12])
# df=pd.concat([outcome6,outcome12])

insert_index = 0
insert_colname = 'Asset'
insert_values = ['Options', 'Options', 'Options', 'Options', 'Options', 'Options',
                 'Options']  # this can be a numpy array too
# insert_values = ['Options','Options'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 1
insert_colname = 'Product'
insert_values = [sheet, sheet, sheet, sheet, sheet, sheet, sheet]  # this can be a numpy array too
# insert_values = [sheet,sheet] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 2
insert_colname = 'Month Ahead'
insert_values = ['1', '2', '3', '4', '5', '6', '12']  # this can be a numpy array too
# insert_values = ['6','12'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

globals()[sheet] = df

import matplotlib.pyplot as mp

df.plot(title=sheet, x='Month Ahead', y=["Prediction", "95%CI.Min", "95%CI.Max"], kind="line", figsize=(10, 10))
mp.show()


#!/usr/bin/env python
# coding: utf-8

# ## Product Selection

# In[1]:
# sheet = 'C1'
sheet='C2'
# sheet='BYX'
# sheet='BZX'
# sheet='EDGA'
# sheet='EDGX_Equities'
# sheet='EXO_EDGX Options'
# sheet='OPT_BZX BATS'
# sheet='CFE'

#C1 Prediction 1m

#load ensemble
import os
from keras.models import load_model
from os import path

# ## 1m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '1','model_corr.h5'))


df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)

values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())

# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble, model


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome1 = pd.DataFrame(outcome)
print(outcome1)

# In[3]:




# ## 2m

# In[4]:


from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '2','model_corr.h5'))

# sheet='BYX'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome2 = pd.DataFrame(outcome)
print('C2_2m')
print(outcome2)


# ## 3m

# In[5]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '3','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome3 = pd.DataFrame(outcome)
print('outcome3')
print(outcome3)


# ## 4m

# In[6]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '4','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
# #
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome4 = pd.DataFrame(outcome)
print(outcome4)


# ## 5m

# In[7]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '5','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 5
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome5 = pd.DataFrame(outcome)
print(outcome5)



# ## 6m

# In[12]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'
ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '6','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome6 = pd.DataFrame(outcome)
print(outcome6)


# ## 12m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '12','model_corr.h5'))

# sheet='OPT_BZX BATS'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
# n_months = 11
n_months = 23
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 15
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1) // 2) * n_features  # the following 12
# n_obs = (n_months+1)* n_features #the following
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1) // 2, n_features))
# train_X = train_X.reshape((train_X.shape[0], (n_months+1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1) // 2, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome12 = pd.DataFrame(outcome)
print(outcome12)


# ## Prediciton Chart

# In[14]:


# Concatenate all dataframes by identical columns

df = pd.concat([outcome1, outcome2, outcome3, outcome4, outcome5, outcome6, outcome12])
# df=pd.concat([outcome6,outcome12])

insert_index = 0
insert_colname = 'Asset'
insert_values = ['Options', 'Options', 'Options', 'Options', 'Options', 'Options',
                 'Options']  # this can be a numpy array too
# insert_values = ['Options','Options'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 1
insert_colname = 'Product'
insert_values = [sheet, sheet, sheet, sheet, sheet, sheet, sheet]  # this can be a numpy array too
# insert_values = [sheet,sheet] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 2
insert_colname = 'Month Ahead'
insert_values = ['1', '2', '3', '4', '5', '6', '12']  # this can be a numpy array too
# insert_values = ['6','12'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

globals()[sheet] = df

import matplotlib.pyplot as mp

df.plot(title=sheet, x='Month Ahead', y=["Prediction", "95%CI.Min", "95%CI.Max"], kind="line", figsize=(10, 10))
mp.show()






# ## Product Selection

# In[1]:
# sheet = 'C1'
# sheet='C2'
# sheet='BYX'
# sheet='BZX'
# sheet='EDGA'
# sheet='EDGX_Equities'
sheet='EXO_EDGX_Options'
# sheet='OPT_BZX_BATS'
# sheet='CFE'

#Prediction 1m

#load ensemble
import os
from keras.models import load_model
from os import path

# ## 1m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '1','model_corr.h5'))


df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)

values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())

# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble, model


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome1 = pd.DataFrame(outcome)
print(outcome1)

# In[3]:




# ## 2m

# In[4]:


from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '2','model_corr.h5'))

# sheet='BYX'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome2 = pd.DataFrame(outcome)
print('C2_2m')
print(outcome2)


# ## 3m

# In[5]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '3','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome3 = pd.DataFrame(outcome)
print('outcome3')
print(outcome3)


# ## 4m

# In[6]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '4','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
# #
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome4 = pd.DataFrame(outcome)
print(outcome4)


# ## 5m

# In[7]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '5','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 5
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome5 = pd.DataFrame(outcome)
print(outcome5)



# ## 6m

# In[12]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'
ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '6','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome6 = pd.DataFrame(outcome)
print(outcome6)


# ## 12m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '12','model_corr.h5'))

# sheet='OPT_BZX BATS'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
# n_months = 11
n_months = 23
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 15
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1) // 2) * n_features  # the following 12
# n_obs = (n_months+1)* n_features #the following
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1) // 2, n_features))
# train_X = train_X.reshape((train_X.shape[0], (n_months+1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1) // 2, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome12 = pd.DataFrame(outcome)
print(outcome12)


# ## Prediciton Chart

# In[14]:


# Concatenate all dataframes by identical columns

df = pd.concat([outcome1, outcome2, outcome3, outcome4, outcome5, outcome6, outcome12])
# df=pd.concat([outcome6,outcome12])

insert_index = 0
insert_colname = 'Asset'
insert_values = ['Options', 'Options', 'Options', 'Options', 'Options', 'Options',
                 'Options']  # this can be a numpy array too
# insert_values = ['Options','Options'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 1
insert_colname = 'Product'
insert_values = [sheet, sheet, sheet, sheet, sheet, sheet, sheet]  # this can be a numpy array too
# insert_values = [sheet,sheet] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 2
insert_colname = 'Month Ahead'
insert_values = ['1', '2', '3', '4', '5', '6', '12']  # this can be a numpy array too
# insert_values = ['6','12'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

globals()[sheet] = df

import matplotlib.pyplot as mp

df.plot(title=sheet, x='Month Ahead', y=["Prediction", "95%CI.Min", "95%CI.Max"], kind="line", figsize=(10, 10))
mp.show()


# ## Product Selection

# In[1]:
# sheet = 'C1'
# sheet='C2'
# sheet='EXO_EDGX_Options'
sheet='OPT_BZX_BATS'

# sheet='BYX'
# sheet='BZX'
# sheet='EDGA'
# sheet='EDGX_Equities'

# sheet='CFE'

#Prediction 1m

#load ensemble
import os
from keras.models import load_model
from os import path

# ## 1m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '1','model_corr.h5'))


df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)

values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())

# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble, model


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome1 = pd.DataFrame(outcome)
print(outcome1)

# In[3]:




# ## 2m

# In[4]:


from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '2','model_corr.h5'))

# sheet='BYX'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome2 = pd.DataFrame(outcome)
print('C2_2m')
print(outcome2)


# ## 3m

# In[5]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '3','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome3 = pd.DataFrame(outcome)
print('outcome3')
print(outcome3)


# ## 4m

# In[6]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '4','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
# #
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome4 = pd.DataFrame(outcome)
print(outcome4)


# ## 5m

# In[7]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '5','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 5
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome5 = pd.DataFrame(outcome)
print(outcome5)



# ## 6m

# In[12]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'
ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '6','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome6 = pd.DataFrame(outcome)
print(outcome6)


# ## 12m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '12','model_corr.h5'))

# sheet='OPT_BZX BATS'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
# n_months = 11
n_months = 23
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 15
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1) // 2) * n_features  # the following 12
# n_obs = (n_months+1)* n_features #the following
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1) // 2, n_features))
# train_X = train_X.reshape((train_X.shape[0], (n_months+1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1) // 2, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome12 = pd.DataFrame(outcome)
print(outcome12)


# ## Prediciton Chart

# In[14]:


# Concatenate all dataframes by identical columns

df = pd.concat([outcome1, outcome2, outcome3, outcome4, outcome5, outcome6, outcome12])
# df=pd.concat([outcome6,outcome12])

insert_index = 0
insert_colname = 'Asset'
insert_values = ['Options', 'Options', 'Options', 'Options', 'Options', 'Options',
                 'Options']  # this can be a numpy array too
# insert_values = ['Options','Options'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 1
insert_colname = 'Product'
insert_values = [sheet, sheet, sheet, sheet, sheet, sheet, sheet]  # this can be a numpy array too
# insert_values = [sheet,sheet] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 2
insert_colname = 'Month Ahead'
insert_values = ['1', '2', '3', '4', '5', '6', '12']  # this can be a numpy array too
# insert_values = ['6','12'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

globals()[sheet] = df

import matplotlib.pyplot as mp

df.plot(title=sheet, x='Month Ahead', y=["Prediction", "95%CI.Min", "95%CI.Max"], kind="line", figsize=(10, 10))
mp.show()









# In[1]:
# sheet = 'C1'
# sheet='C2'
# sheet='EXO_EDGX_Options'
# sheet='OPT_BZX_BATS'

sheet='BYX'
# sheet='BZX'
# sheet='EDGA'
# sheet='EDGX_Equities'

# sheet='CFE'

#Prediction 1m

#load ensemble
import os
from keras.models import load_model
from os import path

# ## 1m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '1','model_corr.h5'))


df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)

values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())

# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble, model


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome1 = pd.DataFrame(outcome)
print(outcome1)

# In[3]:




# ## 2m

# In[4]:


from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '2','model_corr.h5'))

# sheet='BYX'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome2 = pd.DataFrame(outcome)
print('C2_2m')
print(outcome2)


# ## 3m

# In[5]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '3','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome3 = pd.DataFrame(outcome)
print('outcome3')
print(outcome3)


# ## 4m

# In[6]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '4','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
# #
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome4 = pd.DataFrame(outcome)
print(outcome4)


# ## 5m

# In[7]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '5','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 5
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome5 = pd.DataFrame(outcome)
print(outcome5)



# ## 6m

# In[12]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'
ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '6','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome6 = pd.DataFrame(outcome)
print(outcome6)


# ## 12m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '12','model_corr.h5'))

# sheet='OPT_BZX BATS'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
# n_months = 11
n_months = 23
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 15
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1) // 2) * n_features  # the following 12
# n_obs = (n_months+1)* n_features #the following
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1) // 2, n_features))
# train_X = train_X.reshape((train_X.shape[0], (n_months+1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1) // 2, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome12 = pd.DataFrame(outcome)
print(outcome12)


# ## Prediciton Chart

# In[14]:


# Concatenate all dataframes by identical columns

df = pd.concat([outcome1, outcome2, outcome3, outcome4, outcome5, outcome6, outcome12])
# df=pd.concat([outcome6,outcome12])

insert_index = 0
insert_colname = 'Asset'
insert_values = ['Equities', 'Equities', 'Equities', 'Equities', 'Equities', 'Equities',
                 'Equities']  # this can be a numpy array too
# insert_values = ['Options','Options'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 1
insert_colname = 'Product'
insert_values = [sheet, sheet, sheet, sheet, sheet, sheet, sheet]  # this can be a numpy array too
# insert_values = [sheet,sheet] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 2
insert_colname = 'Month Ahead'
insert_values = ['1', '2', '3', '4', '5', '6', '12']  # this can be a numpy array too
# insert_values = ['6','12'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

globals()[sheet] = df

import matplotlib.pyplot as mp

df.plot(title=sheet, x='Month Ahead', y=["Prediction", "95%CI.Min", "95%CI.Max"], kind="line", figsize=(10, 10))
mp.show()



# sheet='BYX'
sheet='BZX'
# sheet='EDGA'
# sheet='EDGX_Equities'

# sheet='CFE'

#Prediction 1m

#load ensemble
import os
from keras.models import load_model
from os import path

# ## 1m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '1','model_corr.h5'))


df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)

values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())

# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble, model


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome1 = pd.DataFrame(outcome)
print(outcome1)

# In[3]:




# ## 2m

# In[4]:


from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '2','model_corr.h5'))

# sheet='BYX'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome2 = pd.DataFrame(outcome)
print('C2_2m')
print(outcome2)


# ## 3m

# In[5]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '3','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome3 = pd.DataFrame(outcome)
print('outcome3')
print(outcome3)


# ## 4m

# In[6]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '4','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
# #
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome4 = pd.DataFrame(outcome)
print(outcome4)


# ## 5m

# In[7]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '5','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 5
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome5 = pd.DataFrame(outcome)
print(outcome5)



# ## 6m

# In[12]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'
ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '6','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome6 = pd.DataFrame(outcome)
print(outcome6)


# ## 12m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '12','model_corr.h5'))

# sheet='OPT_BZX BATS'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
# n_months = 11
n_months = 23
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 15
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1) // 2) * n_features  # the following 12
# n_obs = (n_months+1)* n_features #the following
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1) // 2, n_features))
# train_X = train_X.reshape((train_X.shape[0], (n_months+1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1) // 2, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome12 = pd.DataFrame(outcome)
print(outcome12)


# ## Prediciton Chart

# In[14]:


# Concatenate all dataframes by identical columns

df = pd.concat([outcome1, outcome2, outcome3, outcome4, outcome5, outcome6, outcome12])
# df=pd.concat([outcome6,outcome12])

insert_index = 0
insert_colname = 'Asset'
insert_values = ['Equities', 'Equities', 'Equities', 'Equities', 'Equities', 'Equities',
                 'Equities']  # this can be a numpy array too
# insert_values = ['Options','Options'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 1
insert_colname = 'Product'
insert_values = [sheet, sheet, sheet, sheet, sheet, sheet, sheet]  # this can be a numpy array too
# insert_values = [sheet,sheet] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 2
insert_colname = 'Month Ahead'
insert_values = ['1', '2', '3', '4', '5', '6', '12']  # this can be a numpy array too
# insert_values = ['6','12'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

globals()[sheet] = df

import matplotlib.pyplot as mp

df.plot(title=sheet, x='Month Ahead', y=["Prediction", "95%CI.Min", "95%CI.Max"], kind="line", figsize=(10, 10))
mp.show()




# sheet='BYX'
# sheet='BZX'
sheet='EDGA'
# sheet='EDGX_Equities'

# sheet='CFE'

#Prediction 1m

#load ensemble
import os
from keras.models import load_model
from os import path

# ## 1m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '1','model_corr.h5'))


df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)

values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())

# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble, model


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome1 = pd.DataFrame(outcome)
print(outcome1)

# In[3]:

# ## 2m

# In[4]:


from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '2','model_corr.h5'))

# sheet='BYX'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome2 = pd.DataFrame(outcome)
print('C2_2m')
print(outcome2)


# ## 3m

# In[5]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '3','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome3 = pd.DataFrame(outcome)
print('outcome3')
print(outcome3)


# ## 4m

# In[6]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '4','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
# #
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome4 = pd.DataFrame(outcome)
print(outcome4)


# ## 5m

# In[7]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '5','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 5
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome5 = pd.DataFrame(outcome)
print(outcome5)



# ## 6m

# In[12]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'
ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '6','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome6 = pd.DataFrame(outcome)
print(outcome6)


# ## 12m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '12','model_corr.h5'))

# sheet='OPT_BZX BATS'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
# n_months = 11
n_months = 23
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 15
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1) // 2) * n_features  # the following 12
# n_obs = (n_months+1)* n_features #the following
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1) // 2, n_features))
# train_X = train_X.reshape((train_X.shape[0], (n_months+1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1) // 2, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome12 = pd.DataFrame(outcome)
print(outcome12)


# ## Prediciton Chart

# In[14]:


# Concatenate all dataframes by identical columns

df = pd.concat([outcome1, outcome2, outcome3, outcome4, outcome5, outcome6, outcome12])
# df=pd.concat([outcome6,outcome12])

insert_index = 0
insert_colname = 'Asset'
insert_values = ['Equities', 'Equities', 'Equities', 'Equities', 'Equities', 'Equities',
                 'Equities']  # this can be a numpy array too
# insert_values = ['Options','Options'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 1
insert_colname = 'Product'
insert_values = [sheet, sheet, sheet, sheet, sheet, sheet, sheet]  # this can be a numpy array too
# insert_values = [sheet,sheet] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 2
insert_colname = 'Month Ahead'
insert_values = ['1', '2', '3', '4', '5', '6', '12']  # this can be a numpy array too
# insert_values = ['6','12'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

globals()[sheet] = df

import matplotlib.pyplot as mp

df.plot(title=sheet, x='Month Ahead', y=["Prediction", "95%CI.Min", "95%CI.Max"], kind="line", figsize=(10, 10))
mp.show()





# sheet='BYX'
# sheet='BZX'
# sheet='EDGA'
sheet='EDGX_Equities'

# sheet='CFE'

#Prediction 1m

#load ensemble
import os
from keras.models import load_model
from os import path

# ## 1m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '1','model_corr.h5'))


df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)

values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())

# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble, model


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome1 = pd.DataFrame(outcome)
print(outcome1)

# In[3]:

# ## 2m

# In[4]:


from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '2','model_corr.h5'))

# sheet='BYX'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome2 = pd.DataFrame(outcome)
print('C2_2m')
print(outcome2)


# ## 3m

# In[5]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '3','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome3 = pd.DataFrame(outcome)
print('outcome3')
print(outcome3)


# ## 4m

# In[6]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '4','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
# #
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome4 = pd.DataFrame(outcome)
print(outcome4)


# ## 5m

# In[7]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '5','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 5
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome5 = pd.DataFrame(outcome)
print(outcome5)



# ## 6m

# In[12]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'
ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '6','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome6 = pd.DataFrame(outcome)
print(outcome6)


# ## 12m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '12','model_corr.h5'))

# sheet='OPT_BZX BATS'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
# n_months = 11
n_months = 23
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 15
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1) // 2) * n_features  # the following 12
# n_obs = (n_months+1)* n_features #the following
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1) // 2, n_features))
# train_X = train_X.reshape((train_X.shape[0], (n_months+1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1) // 2, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome12 = pd.DataFrame(outcome)
print(outcome12)


# ## Prediciton Chart

# In[14]:


# Concatenate all dataframes by identical columns

df = pd.concat([outcome1, outcome2, outcome3, outcome4, outcome5, outcome6, outcome12])
# df=pd.concat([outcome6,outcome12])

insert_index = 0
insert_colname = 'Asset'
insert_values = ['Equities', 'Equities', 'Equities', 'Equities', 'Equities', 'Equities',
                 'Equities']  # this can be a numpy array too
# insert_values = ['Options','Options'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 1
insert_colname = 'Product'
insert_values = [sheet, sheet, sheet, sheet, sheet, sheet, sheet]  # this can be a numpy array too
# insert_values = [sheet,sheet] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 2
insert_colname = 'Month Ahead'
insert_values = ['1', '2', '3', '4', '5', '6', '12']  # this can be a numpy array too
# insert_values = ['6','12'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

globals()[sheet] = df

import matplotlib.pyplot as mp

df.plot(title=sheet, x='Month Ahead', y=["Prediction", "95%CI.Min", "95%CI.Max"], kind="line", figsize=(10, 10))
mp.show()





# sheet='BYX'
# sheet='BZX'
# sheet='EDGA'
# sheet='EDGX_Equities'

sheet='CFE'

#Prediction 1m

#load ensemble
import os
from keras.models import load_model
from os import path

# ## 1m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '1',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '1','model_corr.h5'))


df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)

values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())

# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 5):-n_features * (1 + 5) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble, model


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome1 = pd.DataFrame(outcome)
print(outcome1)

# In[3]:

# ## 2m

# In[4]:


from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime


ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '2',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '2','model_corr.h5'))

# sheet='BYX'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1)) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 4):-n_features * (1 + 4) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome2 = pd.DataFrame(outcome)
print('C2_2m')
print(outcome2)


# ## 3m

# In[5]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '3',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '3','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 3):-n_features * (1 + 3) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome3 = pd.DataFrame(outcome)
print('outcome3')
print(outcome3)


# ## 4m

# In[6]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '4',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '4','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
# #
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + 2):-n_features * (1 + 2) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome4 = pd.DataFrame(outcome)
print(outcome4)


# ## 5m

# In[7]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '5',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '5','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 5
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome5 = pd.DataFrame(outcome)
print(outcome5)



# ## 6m

# In[12]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

# sheet='BYX'
ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '6',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '6','model_corr.h5'))

df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# encoder = LabelEncoder()
# # # print(values[:,0]) #PortQyt Value
# values[:,0]=encoder.fit_transform(values[:,0])
# print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
n_months = 5
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 36
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = (n_months + 1) * n_features  # the following fourth month
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1), n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome6 = pd.DataFrame(outcome)
print(outcome6)


# ## 12m

# In[2]:


from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import io
import datetime

ensemble=[]
i=0
while os.path.exists(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')):
  ensemble.append(load_model(path.join('Train & Predicition',sheet, '12',f'ensemble_corr_{i}.h5')))
  i+=1
#load model
model=load_model(path.join('Train & Predicition',sheet, '12','model_corr.h5'))

# sheet='OPT_BZX BATS'
df = pd.read_excel(r'Logical Ports Prediction.xlsx', sheet_name=sheet, header=0)
# df= pd.read_excel(r'Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
df.dropna(subset=['Date'], inplace=True)

from datetime import datetime as dt

last_date = df.loc[:, 'Date']
last_date = last_date.iat[-1]
last_date = last_date.strftime('%Y-%m-%d')
last_date

df = df[df['Date'] <= last_date].reset_index(drop=True)
df.tail()
df['Year'] = pd.to_datetime(df['Date']).dt.strftime('%Y')
df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%m')

df.drop(['Date', 'Logical Ports', 'Asset'], axis=1, inplace=True)
first_column = df.pop('Year')
df.insert(0, 'Year', first_column)
second_column = df.pop('Month')
df.insert(1, 'Month', second_column)
first_column = df.pop('PortQyt')
df.insert(0, 'PortQyt', first_column)

df.shape
print(len(list(df)[:]))
# # df=df.drop(['D&J_Close30d_Avg.','D&J_Close30d_Min','Monthly Real GDP Index','TCV','FEDFUNDS','Unemployement Rate','SPX_Price30d_Std.','D&J_Close30d_Std.','VIX_Close30d_Max.','VIX_Close30d_Avg.','VIX_Close30d_Std.','VIX_Close30d_Min.'],axis=1)


values = df.values
#
# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # encoder = LabelEncoder()
# # # # print(values[:,0]) #PortQyt Value
# # values[:,0]=encoder.fit_transform(values[:,0])
# # print(values[:,0])

# import matplotlib.pyplot as pp
# pp.plot(values[:,0])
# pp.show()

# # print(values[:,4].shape)
# # print(values[:,4])
values = values.astype('float32')

# specify the number of lag hours
# n_months = 11
n_months = 23
n_features = len(list(df)[:])

# Normalize the first feature
from sklearn.preprocessing import StandardScaler

scaler2 = MinMaxScaler(feature_range=(0, 1)).fit(values[:, 0:1])
# train_y = scaler2.transform(values[:,-n_features])
# scaled2 = scaler2.fit_transform(values) #try


# # normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# len(list(dataset.columns))-3

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# frame as supervised learning
reframed = series_to_supervised(scaled, n_months, 1)
# print(reframed.shape)


# print(reframed.tail())


m = 6 - 6
# split into train and test sets
values = reframed.values
n_train_months = 15
train = values[:n_train_months, :]
test = values[n_train_months:, :]
print(values.shape)
# split into input and outputs
n_obs = ((n_months + 1) // 2) * n_features  # the following 12
# n_obs = (n_months+1)* n_features #the following
train.shape
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
train_X, train_y = train[:, :n_obs], train[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
test_X, test_y = test[:, :n_obs], test[:, -n_features * (1 + m):-n_features * (1 + m) + 1]
print(test.shape)

# import matplotlib.pyplot as pp
# pp.plot(train_y)
# pp.show()

train_X = train_X.reshape((train_X.shape[0], (n_months + 1) // 2, n_features))
# train_X = train_X.reshape((train_X.shape[0], (n_months+1), n_features))
test_X = test_X.reshape((test_X.shape[0], (n_months + 1) // 2, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# prediction interval for mlps on the housing regression dataset
from numpy import asarray
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import math
import numpy as np


# define and fit the model
def fit_model(X_train, y_train):
    features = X_train.shape[1]
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, verbose=0, epochs=50, batch_size=16)
    return model


# fit an ensemble of models
def fit_ensemble(n_members, X_train, X_test, y_train, y_test):
    ensemble = list()
    for i in range(n_members):
        model = fit_model(X_train, y_train)  # define and fit the model on the training set
        yhat = model.predict(X_test, verbose=0)  # evaluate model on the test set
        print(y_test.shape, yhat.shape)
        mae = mean_absolute_error(y_test, yhat)
        print('>%d, MAE: %.3f' % (i + 1, mae))
        ensemble.append(model)  # store the model
    return ensemble


# n_members = 20
# ensemble = list()
# for i in range(n_members):
#     model = fit_model(train_X, train_y)  # define and fit the model on the training set
#     yhat = model.predict(test_X, verbose=0)  # evaluate model on the test set
#     print(test_y, yhat.shape)
#     mae = mean_absolute_error(test_y, yhat)
#     print('>%d, MAE: %.3f' % (i + 1, mae))
#     ensemble.append(model)  # store the model


# make predictions with the ensemble and calculate a prediction interval
def predict_with_pi(ensemble, X, n_members):
    yhat = [model.predict(X, verbose=0) for model in ensemble]
    yhat = asarray(yhat)
    yhat = np.reshape(yhat, [n_members, 1])
    print(yhat.shape)
    # yhat = scaled2.inverse_transform(yhat)
    yhat = scaler2.inverse_transform(yhat)  # invert scaling for actual
    yhat = yhat[:, 0]
    interval = 1.96 * yhat.std() / math.sqrt(len(yhat))
    lower, upper = yhat.mean() - interval, yhat.mean() + interval
    return lower, yhat.mean(), upper


n_members = len(ensemble)
# make predictions with prediction interval
newX = asarray([test_X[0, :]])
lower, mean, upper = predict_with_pi(ensemble, newX, n_members)
# print(test_y)
# print('Point prediction: %.1f' % mean)
print('95%% prediction interval: [%.1f, %.1f]' % (lower, upper))
print('True value: %.1f' % test_y[0])
outcome = {'Prediction': [mean],
           '95%CI.Min': [lower],
           '95%CI.Max': [upper]
           }

outcome12 = pd.DataFrame(outcome)
print(outcome12)


# ## Prediciton Chart

# In[14]:


# Concatenate all dataframes by identical columns

df = pd.concat([outcome1, outcome2, outcome3, outcome4, outcome5, outcome6, outcome12])
# df=pd.concat([outcome6,outcome12])

insert_index = 0
insert_colname = 'Asset'
insert_values = ['Futures', 'Futures', 'Futures', 'Futures', 'Futures', 'Futures',
                 'Futures']  # this can be a numpy array too
# insert_values = ['Options','Options'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 1
insert_colname = 'Product'
insert_values = [sheet, sheet, sheet, sheet, sheet, sheet, sheet]  # this can be a numpy array too
# insert_values = [sheet,sheet] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

insert_index = 2
insert_colname = 'Month Ahead'
insert_values = ['1', '2', '3', '4', '5', '6', '12']  # this can be a numpy array too
# insert_values = ['6','12'] # this can be a numpy array too
df.insert(loc=insert_index, column=insert_colname, value=insert_values)

globals()[sheet] = df

import matplotlib.pyplot as mp

df.plot(title=sheet, x='Month Ahead', y=["Prediction", "95%CI.Min", "95%CI.Max"], kind="line", figsize=(10, 10))
mp.show()



## ExcelOutput
df=pd.concat([C1, C2,EXO_EDGX_Options,OPT_BZX_BATS,BYX,BZX,EDGA,EDGX_Equities,CFE])
print(df)


# In[19]:


from datetime import datetime as dt
last_date= pd.read_excel('Logical Ports Prediction.xlsx',sheet_name='C2',header=0)
last_date.dropna(subset=['Date'],inplace=True)
last_date.tail()
last_date=last_date.loc[:,'Date']
last_date=last_date.iat[-1]
last_date=last_date.strftime('%m-%y')
last_date
# # .strftime('%y-%m-%d')


# In[22]:


with pd.ExcelWriter('Logical Ports Prediction.xlsx',mode='a',engine='openpyxl',if_sheet_exists='replace') as writer:
    df.to_excel(writer,sheet_name=last_date)