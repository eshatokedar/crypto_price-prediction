#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler
import pickle
import nbformat
from nbconvert import PythonExporter

# Load dataset
maindf = pd.read_csv('BTC-USD (1).csv')
maindf['Date'] = pd.to_datetime(maindf['Date'], format='%Y-%m-%d')

# Print dataset information
print('Total number of days present in the dataset: ', maindf.shape[0])
print('Total number of fields present in the dataset: ', maindf.shape[1])
print('Null Values:', maindf.isnull().values.sum())
print('NA values:', maindf.isnull().values.any())
print('Starting Date', maindf.iloc[0]['Date'])
print('Ending Date', maindf.iloc[-1]['Date'])

# Example data preprocessing for 2014
y_2014 = maindf[(maindf['Date'] >= '2014-09-17') & (maindf['Date'] < '2014-12-31')]
y_2014 = y_2014.drop(['Adj Close', 'Volume'], axis=1)

# Month-wise analysis for 2014
monthvise = y_2014.groupby(y_2014['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
monthvise = monthvise.reindex(new_order, axis=0)

# Plotting month-wise comparison
fig = go.Figure()
fig.add_trace(go.Bar(x=monthvise.index, y=monthvise['Open'], name='Bitcoin Open Price', marker_color='crimson'))
fig.add_trace(go.Bar(x=monthvise.index, y=monthvise['Close'], name='Bitcoin Close Price', marker_color='lightsalmon'))
fig.update_layout(barmode='group', xaxis_tickangle=-45, title='Monthwise comparision between Bitcoin open and close price')
fig.show()

# High and Low prices for 2014
monthvise_high = y_2014.groupby(y_2014['Date'].dt.strftime('%B'))['High'].max().reindex(new_order, axis=0)
monthvise_low = y_2014.groupby(y_2014['Date'].dt.strftime('%B'))['Low'].min().reindex(new_order, axis=0)

fig = go.Figure()
fig.add_trace(go.Bar(x=monthvise_high.index, y=monthvise_high, name='Bitcoin high Price', marker_color='rgb(0, 153, 204)'))
fig.add_trace(go.Bar(x=monthvise_low.index, y=monthvise_low, name='Bitcoin low Price', marker_color='rgb(255, 128, 0)'))
fig.update_layout(barmode='group', title=' Monthwise High and Low Bitcoin price')
fig.show()

# Line plot for 2014
names = cycle(['Bitcoin Open Price', 'Bitcoin Close Price', 'Bitcoin High Price', 'Bitcoin Low Price'])
fig = px.line(y_2014, x=y_2014.Date, y=[y_2014['Open'], y_2014['Close'], y_2014['High'], y_2014['Low']],
              labels={'Date': 'Date', 'value': 'Bitcoin value'})
fig.update_layout(title_text='Bitcoin analysis chart', font_size=15, font_color='black', legend_title_text='Bitcoin Parameters')
fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Preprocessing data for 2015
y_2015 = maindf[(maindf['Date'] >= '2015-01-01') & (maindf['Date'] < '2016-01-01')]
y_2015 = y_2015.drop(['Adj Close', 'Volume'], axis=1)

# Month-wise analysis for 2015
monthvise = y_2015.groupby(y_2015['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
monthvise = monthvise.reindex(new_order, axis=0)

# Prepare data for training
closedf = maindf[['Date', 'Close']]
closedf = closedf[closedf['Date'] > '2021-02-19']
print("Total data for prediction: ", closedf.shape[0])

# Plotting close prices
fig = px.line(closedf, x=closedf.Date, y=closedf.Close, labels={'date': 'Date', 'close': 'Close Stock'})
fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
fig.update_layout(title_text='Whole period of timeframe of Bitcoin close price 2014-2022', plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Normalize and prepare training and test data
del closedf['Date']
scaler = MinMaxScaler(feature_range=(0, 1))
closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))
training_size = int(len(closedf) * 0.60)
test_size = len(closedf) - training_size
train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :1]

# Create dataset for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and train the model
model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, verbose=1)

# Plot training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.show()

# Predictions and evaluation metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluation metrics
print("Train data RMSE: ", math.sqrt(mean_squared_error(original_ytrain, train_predict)))
print("Train data MSE: ", mean_squared_error(original_ytrain, train_predict))
print("Train data MAE: ", mean_absolute_error(original_ytrain, train_predict))
print("-------------------------------------------------------------------------------------")
print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest, test_predict)))
print("Test data MSE: ", mean_squared_error(original_ytest, test_predict))
print("Test data MAE: ", mean_absolute_error(original_ytest, test_predict))
print("Train data explained variance regression score:", explained_variance_score(original_ytrain, train_predict))
print("Test data explained variance regression score:", explained_variance_score(original_ytest, test_predict))
print("Train data R2 score:", r2_score(original_ytrain, train_predict))
print("Test data R2 score:", r2_score(original_ytest, test_predict))

# Plot predictions
look_back = time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(closedf) - 1, :] = test_predict

plotdf = pd.DataFrame({'date': closedf, 'original_close': closedf.flatten(), 'train_predicted_close': trainPredictPlot.reshape(1, -1)[0], 'test_predicted_close': testPredictPlot.reshape(1, -1)[0]})
names = cycle(['Original close price', 'Train predicted close price', 'Test predicted close price'])
fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'], plotdf['test_predicted_close']],
              labels={'value': 'Stock price', 'date': 'Date'})
fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Predict future values
x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
temp_input = x_input.flatten().tolist()
lst_output = []
n_steps = time_step
pred_days = 30
while len(lst_output) < pred_days:
    if len(temp_input) > time_step:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
    else:
        x_input = x_input.reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())

print("Output of predicted next days: ", len(lst_output))

# Plot future predictions
last_days = np.arange(1, time_step + 1)
day_pred = np.arange(time_step + 1, time_step + pred_days + 1)
temp_mat = np.empty((len(last_days) + pred_days + 1, 1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1, -1).tolist()[0]
last_original_days_value = temp_mat
next_predicted_days_value = temp_mat
last_original_days_value[0:time_step + 1] = scaler.inverse_transform(closedf[len(closedf) - time_step:]).reshape(1, -1).tolist()[0]
next_predicted_days_value[time_step + 1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).reshape(1, -1).tolist()[0]
new_pred_plot = pd.DataFrame({'last_original_days_value': last_original_days_value, 'next_predicted_days_value': next_predicted_days_value})

names = cycle(['Last 15 days close price', 'Predicted next 30 days close price'])
fig = px.line(new_pred_plot, x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'], new_pred_plot['next_predicted_days_value']],
              labels={'value': 'Stock price', 'index': 'Timestamp'})
fig.update_layout(title_text='Compare last 15 days vs next 30 days',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Save the model
os.makedirs("model", exist_ok=True)
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved successfully.")
model.save('model.h5')
print("Model saved using Keras method.")

# Convert the notebook to a Python script
with open('crypto_price_prediction.ipynb') as f:
    notebook = nbformat.read(f, as_version=4)
exporter = PythonExporter()
script, _ = exporter.from_notebook_node(notebook)
with open('crypto_price_prediction.py', 'w') as f:
    f.write(script)
print("Conversion complete: 'crypto_price_prediction.py' has been created.")
