import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

def lstm_predict(df, price_type="Close", forecast_days=30):
    """Direct price prediction using past N days to predict next price"""
    try:
        if price_type not in df.columns:
            raise ValueError(f"Column '{price_type}' not found")
        
        prices = df[[price_type]].values.flatten()
        
        if len(prices) < 20:
            raise ValueError(f"Need at least 20 data points, got {len(prices)}")
        
        prices = np.nan_to_num(prices, nan=np.nanmean(prices[~np.isnan(prices)]))
        
        print(f"📊 Data: {len(prices)} price points ({df.index[0].date()} to {df.index[-1].date()})")
        
        # Use MinMaxScaler for proper normalization (0 to 1 range)
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Create sliding window dataset
        seq_len = 10  # Use last 10 days to predict next day
        X = []
        y = []
        
        for i in range(len(prices_scaled) - seq_len):
            X.append(prices_scaled[i:i+seq_len])
            y.append(prices_scaled[i+seq_len])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        if len(X) < 10:
            raise ValueError(f"Not enough samples: {len(X)}")
        
        print(f"✅ Created {len(X)} training samples (seq_len={seq_len})")
        
        # Train/test split
        split = int(len(X) * 0.85)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train models
        models = {
            'GB': GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.9, random_state=42
            ),
            'RF': RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            'SVR': SVR(kernel='rbf', C=100, epsilon=0.01, gamma='scale'),
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test) if len(X_test) > 0 else 0
            print(f"  {name}: Train R²={train_score:.4f}, Test R²={test_score:.4f}")
        
        # Predict on all training data for metrics
        gb_pred = models['GB'].predict(X)
        rf_pred = models['RF'].predict(X)
        svr_pred = models['SVR'].predict(X)
        
        # Weighted ensemble (GB is most reliable)
        ensemble_pred = 0.5 * gb_pred + 0.3 * rf_pred + 0.2 * svr_pred
        
        # Inverse transform to get actual prices
        predictions_actual = scaler.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
        actual_prices = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual_prices, predictions_actual))
        mape = mean_absolute_percentage_error(actual_prices, predictions_actual) * 100
        
        print(f"✅ Model Accuracy - RMSE: ${rmse:.2f}, MAPE: {mape:.2f}%")
        
        # Forecast next forecast_days
        forecast = []
        last_sequence = prices_scaled[-seq_len:].copy()
        
        for _ in range(forecast_days):
            # Predict next price (normalized)
            next_pred_norm = (
                0.5 * models['GB'].predict(last_sequence.reshape(1, -1))[0] +
                0.3 * models['RF'].predict(last_sequence.reshape(1, -1))[0] +
                0.2 * models['SVR'].predict(last_sequence.reshape(1, -1))[0]
            )
            
            # Clip to reasonable range
            next_pred_norm = np.clip(next_pred_norm, 0, 1)
            
            # Convert back to actual price
            next_price = scaler.inverse_transform([[next_pred_norm]])[0, 0]
            forecast.append(next_price)
            
            # Update sequence
            last_sequence = np.concatenate([last_sequence[1:], [next_pred_norm]])
        
        forecast = np.array(forecast)
        
        return actual_prices, predictions_actual, forecast, rmse, mape
    
    except Exception as e:
        print(f"❌ Error in lstm_predict: {e}")
        import traceback
        traceback.print_exc()
        raise
