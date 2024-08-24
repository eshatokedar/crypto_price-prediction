import os
import pandas as pd
import numpy as np
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle
import pickle
import nbformat
from nbconvert import PythonExporter

# Load dataset
maindf = pd.read_csv('BTC-USD (1).csv')
maindf['Date'] = pd.to_datetime(maindf['Date'], format='%Y-%m-%d')

# Preprocess data (Example for 2014)
y_2014 = maindf[(maindf['Date'] >= '2014-09-17') & (maindf['Date'] < '2014-12-31')]
y_2014 = y_2014.drop(['Adj Close', 'Volume'], axis=1)

# Visualization
monthvise = y_2014.groupby(y_2014['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
monthvise = monthvise.reindex(new_order, axis=0)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=monthvise.index,
    y=monthvise['Open'],
    name='Bitcoin Open Price',
    marker_color='crimson'
))
fig.add_trace(go.Bar(
    x=monthvise.index,
    y=monthvise['Close'],
    name='Bitcoin Close Price',
    marker_color='lightsalmon'
))
fig.update_layout(barmode='group', xaxis_tickangle=-45, 
                  title='Monthwise comparison between Bitcoin open and close price')
fig.show()

monthvise_high = y_2014.groupby(y_2014['Date'].dt.strftime('%B'))['High'].max()
monthvise_high = monthvise_high.reindex(new_order, axis=0)
monthvise_low = y_2014.groupby(y_2014['Date'].dt.strftime('%B'))['Low'].min()
monthvise_low = monthvise_low.reindex(new_order, axis=0)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=monthvise_high.index,
    y=monthvise_high,
    name='Bitcoin High Price',
    marker_color='rgb(0, 153, 204)'
))
fig.add_trace(go.Bar(
    x=monthvise_low.index,
    y=monthvise_low,
    name='Bitcoin Low Price',
    marker_color='rgb(255, 128, 0)'
))
fig.update_layout(barmode='group', 
                  title='Monthwise High and Low Bitcoin price')
fig.show()

names = cycle(['Bitcoin Open Price', 'Bitcoin Close Price', 'Bitcoin High Price', 'Bitcoin Low Price'])
fig = px.line(y_2014, x=y_2014.Date, y=[y_2014['Open'], y_2014['Close'], y_2014['High'], y_2014['Low']],
             labels={'Date': 'Date', 'value': 'Bitcoin value'})
fig.update_layout(title_text='Bitcoin analysis chart', font_size=15, font_color='black', legend_title_text='Bitcoin Parameters')
fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Further analysis for 2015
y_2015 = maindf[(maindf['Date'] >= '2015-01-01') & (maindf['Date'] < '2016-01-01')]
y_2015 = y_2015.drop(['Adj Close', 'Volume'], axis=1)
monthvise = y_2015.groupby(y_2015['Date'].dt.strftime('%B'))[['Open', 'Close']].mean()
monthvise = monthvise.reindex(new_order, axis=0)

# Prepare data for modeling
closedf = maindf[['Date', 'Close']]
closedf = closedf[closedf['Date'] > '2021-02-19']
del closedf['Date']
scaler = MinMaxScaler(feature_range=(0, 1))
closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

training_size = int(len(closedf) * 0.60)
test_size = len(closedf) - training_size
train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :1]

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

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(10, input_shape=(None, 1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, verbose=1)

# Save and load the model
model.save('model.h5')
model = load_model('model.h5')

# Predictions and performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
original_ytest = scaler.inverse_transform(y_test.reshape(-1, 1))

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

# Prepare data for plotting predictions
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict) + time_step, :] = train_predict
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (time_step * 2) + 1:len(closedf) - 1, :] = test_predict

trainPredictPlot = trainPredictPlot.flatten()
testPredictPlot = testPredictPlot.flatten()
closedf_flatten = closedf.flatten()

plotdf = pd.DataFrame({
    'date': np.arange(len(closedf)),  # assuming a simple range for dates
    'original_close': closedf_flatten,
    'train_predicted_close': trainPredictPlot,
    'test_predicted_close': testPredictPlot
})

names = cycle(['Original close price', 'Train predicted close price', 'Test predicted close price'])
fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original_close'], plotdf['train_predicted_close'], plotdf['test_predicted_close']],
              labels={'value': 'Stock price', 'date': 'Date'})
fig.update_layout(title_text='Comparison between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t: t.update(name=next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Predict future values
x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
temp_input = list(x_input[0])
lst_output = []
n_steps = time_step
pred_days = 30

while len(lst_output) < pred_days:
    if len(temp_input) > time_step:
        x_input = np.array(temp_input[1:]).reshape(1, n_steps, 1)
    else:
        x_input = np.array(temp_input).reshape(1, n_steps, 1)
    yhat = model.predict(x_input, verbose=0)
    temp_input.extend(yhat[0].tolist())
    temp_input = temp_input[1:]
    lst_output.append(yhat[0][0])

x_input = scaler.inverse_transform(np.array(temp_input).reshape(-1, 1))
future_dates = [dt.datetime.now() + dt.timedelta(days=x) for x in range(1, pred_days + 1)]

# Prepare future prediction DataFrame
future_df = pd.DataFrame({
    'date': future_dates,
    'predicted_close': scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten()
})

fig = px.line(future_df, x='date', y='predicted_close', title='Future Bitcoin Price Predictions')
fig.update_layout(title_text='Future Bitcoin Price Predictions',
                  plot_bgcolor='white', font_size=15, font_color='black')
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()
