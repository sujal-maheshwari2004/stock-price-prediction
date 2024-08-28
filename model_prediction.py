import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

def make_predictions(model_path, data_test_scaled):
    model = load_model(model_path)
    
    x = []
    y = []

    for i in range(100, data_test_scaled.shape[0]):
        x.append(data_test_scaled[i-100:i])
        y.append(data_test_scaled[i,0])

    x, y = np.array(x), np.array(y)
    predict = model.predict(x)

    return x, y, predict

def scale_predictions(predict, y, scaler):
    scale = 1 / scaler.scale_
    predict = predict * scale
    y = y * scale

    return predict, y

def calculate_error_metrics(y, predict):
    mse = mean_squared_error(y, predict)
    mae = mean_absolute_error(y, predict)
    
    return mse, mae
