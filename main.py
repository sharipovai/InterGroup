import requests
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

def load_data_and_save():
    base = 'https://api.binance.com'
    path = '/api/v3/klines'
    url = base + path
    param = {'symbol': 'ETHUSDT', 'interval': '5m', 'limit': 1000}
    responce = requests.get(url, params = param)
    if responce.status_code == 200:
        data = pd.DataFrame(responce.json())
        new_data = pd.DataFrame()
        new_data['time'] = data.iloc[:,0]
        new_data['price'] = data.iloc[:, 1]
        new_data.to_csv("data.csv", index=False)
    else:
        print("Error")

def read_data():
    data = pd.read_csv('data.csv')
    return data


def trim_dataset(mat, batch_size):
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

def prediction(data):
    first, train, test = np.split(data, [0, int(0.7*(len(data)))])
    train_x = train["time"]
    train_y = train["price"]
    test_x = test["time"]
    test_y = test["price"]

    train_x = np.array(train_x)
    scaler = StandardScaler()
    train_x = train_x.reshape(-1, 1)
    train_x = scaler.fit_transform(train_x)

    test_x = np.array(test_x)
    test_x_sc = test_x.reshape(-1, 1)
    test_x_sc = scaler.fit_transform(test_x_sc)

    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (train_x.shape[0], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    model.fit(trim_dataset(train_x, 32), trim_dataset(train_y, 32), epochs = 100, batch_size = 32)
    predicted_price = model.predict(test_x_sc)

    out = []
    for elem in predicted_price:
        out.append(elem[0][0])

    pd_test_x = pd.to_datetime(test_x, unit="ms")
    test_x_plt = list(pd_test_x)
    test_y_plt = list(test_y)

    mse = mean_squared_error(test_y_plt, out)
    print('MSE = ', mse)

    plt.plot(test_x_plt, test_y_plt, color = 'red', label = 'Real Price')
    plt.plot(test_x_plt, out, color = 'blue', label = 'Predicted Price')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

#load_data_and_save()
data = read_data()
prediction(data)