import os
import pandas as pd
import numpy as np
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from tensorflow.keras.models import load_model
import nbformat
from nbconvert import PythonExporter

# Load dataset
maindf = pd.read_csv('BTC-USD (1).csv')
maindf['Date'] = pd.to_datetime(maindf['Date'], format='%Y-%m-%d')

# Preprocess data (Example for 2014)
y_2014 = maindf[(maindf['Date'] >= '2014-09-17') & (maindf['Date'] < '2014-12-31')]
y_2014 = y_2014.drop(['Adj Close', 'Volume'], axis=1)

# Create model and train it
closedf = maindf[['Date', 'Close']]
closedf = closedf[closedf['Date'] > '2021-02-19']
del closedf['Date']
scaler = MinMaxScaler(feature_range=(0, 1))
closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

training_size = int(len(closedf) * 0.60)
test_size = len(closedf) - training_size
train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :1]

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

# Save the model using TensorFlow/Keras
model.save('model.h5')

# Load the model to ensure it was saved correctly
model = load_model('model.h5')

# Convert notebook to Python script
try:
    with open('crypto_price_prediction.ipynb') as f:
        notebook = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(notebook)
    with open('crypto_price_prediction.py', 'w') as f:
        f.write(script)
    print("Conversion complete: 'crypto_price_prediction.py' has been created.")
except Exception as e:
    print(f"Error converting notebook: {e}")




