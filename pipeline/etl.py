import pandas as pd
import numpy as np

def clean_stock_data(df):
    """Clean raw stock data: fill missing values, remove duplicates, sort by date."""
    df = df.copy()
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    df = df.ffill().bfill()  # Forward fill then backward fill
    return df

def add_features(df):
    """Add technical indicators: MA20, MA50, RSI, MACD, Bollinger Bands, ATR, Volatility"""
    df = df.copy()
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # High-Low Spread
    df['HL_Spread'] = df['High'] - df['Low']
    
    # RSI (Relative Strength Index)
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'], df['Signal_Line'], df['MACD_Histogram'] = calculate_macd(df['Close'])
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    # ATR (Average True Range) - for volatility
    df['ATR'] = calculate_atr(df, period=14)
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # Volatility (20-day rolling standard deviation of returns)
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    df = df.bfill()  # Backward fill for any remaining NaNs
    return df

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal Line"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    middle_band = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

