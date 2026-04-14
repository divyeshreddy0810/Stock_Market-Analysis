import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def linear_regression_predict(df, price_type="Close", forecast_days=30):

    try:
        if price_type not in df.columns:
            raise ValueError(f"Column '{price_type}' not found")
        
        prices = df[[price_type]].values.flatten()
        
        if len(prices) < 20:
            raise ValueError(f"Need at least 20 data points, got {len(prices)}")
        
        prices = np.nan_to_num(prices, nan=np.nanmean(prices[~np.isnan(prices)]))
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        seq_len = 10 
        X = []
        y = []
        
        for i in range(len(prices_scaled) - seq_len):
            X.append(prices_scaled[i:i+seq_len])
            y.append(prices_scaled[i+seq_len])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        if len(X) < 10:
            raise ValueError(f"Not enough samples: {len(X)}")
            
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future steps
        last_sequence = prices_scaled[-seq_len:].copy()
        predictions_scaled = []
        
        for _ in range(forecast_days):
            pred = model.predict(last_sequence.reshape(1, -1))[0]
            predictions_scaled.append(pred)
            last_sequence = np.append(last_sequence[1:], pred)
            
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        
        # Calculate trailing metrics for context
        y_pred = model.predict(X)
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        
        return {
            "predictions": predictions.tolist(),
            "rmse": float(rmse),
            "message": "Success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": "Prediction failed"
        }
