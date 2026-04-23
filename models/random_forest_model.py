import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

def random_forest_predict(df, price_type="Close", forecast_days=30, evaluation_df=None):

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
            
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        if evaluation_df is None:
            evaluation_df = df

        if price_type not in evaluation_df.columns:
            raise ValueError(f"Column '{price_type}' not found in evaluation data")

        eval_prices = evaluation_df[[price_type]].values.flatten()
        if len(eval_prices) <= seq_len:
            raise ValueError(f"Need more than {seq_len} evaluation points, got {len(eval_prices)}")

        eval_prices = np.nan_to_num(eval_prices, nan=np.nanmean(eval_prices[~np.isnan(eval_prices)]))
        eval_scaled = scaler.transform(eval_prices.reshape(-1, 1)).flatten()

        X_eval = []
        y_eval = []
        for i in range(len(eval_scaled) - seq_len):
            X_eval.append(eval_scaled[i:i + seq_len])
            y_eval.append(eval_scaled[i + seq_len])

        X_eval = np.array(X_eval, dtype=np.float32)
        y_eval = np.array(y_eval, dtype=np.float32)
        if len(X_eval) == 0:
            raise ValueError("No evaluation samples were created")

        # Historical predictions for evaluation and charting.
        y_pred = model.predict(X_eval)
        actual_prices = scaler.inverse_transform(y_eval.reshape(-1, 1)).flatten()
        predictions_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        rmse = np.sqrt(mean_squared_error(actual_prices, predictions_actual))
        mape = mean_absolute_percentage_error(actual_prices, predictions_actual) * 100
        
        # Predict future steps
        last_sequence = eval_scaled[-seq_len:].copy()
        predictions_scaled = []
        
        for _ in range(forecast_days):
            pred = model.predict(last_sequence.reshape(1, -1))[0]
            predictions_scaled.append(pred)
            last_sequence = np.append(last_sequence[1:], pred)
            
        forecast = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        
        return actual_prices, predictions_actual, forecast, float(rmse), float(mape)
        
    except Exception as e:
        raise ValueError(f"Random Forest prediction failed: {str(e)}")
