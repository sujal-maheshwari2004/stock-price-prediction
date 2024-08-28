import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import save_model

def prepare_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start_date, end_date)
    data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_train)
    
    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scaled = scaler.transform(data_test)
    
    return data, data_train, data_test, scaler, data_test_scaled

def create_and_train_model(data_test_scaled):
    x = []
    y = []

    for i in range(100, data_test_scaled.shape[0]):
        x.append(data_test_scaled[i-100:i])
        y.append(data_test_scaled[i,0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x, y, epochs=10, batch_size=32)
    
    return model

def save_model_to_file(model, filename):
    model.save(filename)
