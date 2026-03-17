import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(prices, look_back=60):
    """Scale prices and create sequences for LSTM"""
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler
