import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to create dataset for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Function to normalize data
def normalize_data(data, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = np.array(data).reshape(-1, 1)
        data = scaler.fit_transform(data)
    else:
        data = np.array(data).reshape(-1, 1)
        data = scaler.transform(data)
    return data, scaler

# Function to predict prices
def predict_crypto_price(data, model, time_step=15, pred_days=30):
    # Normalize the input data
    data_normalized, scaler = normalize_data(data)
    
    # Prepare the input for LSTM model
    x_input = data_normalized[-time_step:].reshape(1, -1, 1)
    temp_input = x_input.flatten().tolist()

    lst_output = []
    while len(lst_output) < pred_days:
        x_input = np.array(temp_input[-time_step:]).reshape(1, -1, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat.flatten().tolist())
        lst_output.extend(yhat.flatten().tolist())
    
    # Inverse transform the predicted values
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.array(data).reshape(-1, 1))
    predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))
    return predictions

# Streamlit app
st.title('Cryptocurrency Price Prediction')

# Upload a CSV file with historical price data
uploaded_file = st.file_uploader("Upload a CSV file with historical prices", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.head())

    # Assuming the CSV has a 'Close' column
    if 'Close' in df.columns and 'Date' in df.columns:
        data = df['Close'].values
        last_date = pd.to_datetime(df['Date']).max()
        
        # Predict future prices
        predictions = predict_crypto_price(data, model)

        # Create a DataFrame for the results
        future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, len(predictions) + 1)]
        result_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close Price': predictions.flatten()
        })

        st.write("Prediction Results:", result_df)
        st.line_chart(result_df.set_index('Date'))
    else:
        st.error("CSV file must contain 'Close' and 'Date' columns.")
