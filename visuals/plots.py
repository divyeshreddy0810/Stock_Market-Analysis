import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def plot_compare(data):
    fig, ax = plt.subplots(figsize=(10,5))
    for t, df in data.items():
        ax.plot(df.index, df["Close"], label=t)
    ax.set_title("Stock Price Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

def plot_prediction(ticker, actual, pred, price_type, forecast=None, forecast_days=0):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(actual, label=f"Actual {price_type}")
    ax.plot(pred, label=f"Predicted {price_type}")
    if forecast is not None:
        ax.plot(range(len(actual), len(actual)+forecast_days), forecast,
                label=f"Forecast {price_type}", linestyle='--')
    ax.set_title(f"{ticker} {price_type} Price Prediction & Forecast")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

def plot_normalized_comparison(data_dict, title="Normalized Price Comparison"):
    """Plot normalized prices for multiple stocks (0-1 scale)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for ticker, df in data_dict.items():
        prices = df['Close'].values
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        ax.plot(df.index, normalized, label=ticker, linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price (0-1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_performance_comparison(data_dict, title="Performance Comparison (%)"):
    """Plot percentage returns from start date"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for ticker, df in data_dict.items():
        returns = ((df['Close'] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100)
        ax.plot(df.index, returns, label=ticker, linewidth=2, marker='o', markersize=3)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_volatility_comparison(data_dict, title="30-Day Rolling Volatility"):
    """Plot volatility (rolling standard deviation of returns)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for ticker, df in data_dict.items():
        if 'Volatility' in df.columns:
            ax.plot(df.index, df['Volatility'], label=ticker, linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_rsi_comparison(data_dict, title="RSI (Relative Strength Index) Comparison"):
    """Plot RSI indicator for multiple stocks"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for ticker, df in data_dict.items():
        if 'RSI' in df.columns:
            ax.plot(df.index, df['RSI'], label=ticker, linewidth=2)
    
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_macd_comparison(data_dict, title="MACD (Moving Average Convergence Divergence)"):
    """Plot MACD and Signal Line for multiple stocks in subplots"""
    tickers = list(data_dict.keys())
    n_tickers = len(tickers)
    
    fig, axes = plt.subplots(n_tickers, 1, figsize=(12, 4*n_tickers))
    if n_tickers == 1:
        axes = [axes]
    
    for idx, (ticker, df) in enumerate(data_dict.items()):
        if 'MACD' in df.columns:
            ax = axes[idx]
            ax.plot(df.index, df['MACD'], label='MACD', linewidth=2)
            ax.plot(df.index, df['Signal_Line'], label='Signal Line', linewidth=2)
            ax.bar(df.index, df['MACD_Histogram'], label='Histogram', alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(f"{ticker} - {title}")
            ax.set_ylabel("MACD")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_bollinger_bands_comparison(data_dict, title="Bollinger Bands"):
    """Plot Bollinger Bands for multiple stocks in subplots"""
    tickers = list(data_dict.keys())
    n_tickers = len(tickers)
    
    fig, axes = plt.subplots(n_tickers, 1, figsize=(12, 4*n_tickers))
    if n_tickers == 1:
        axes = [axes]
    
    for idx, (ticker, df) in enumerate(data_dict.items()):
        if 'BB_Upper' in df.columns:
            ax = axes[idx]
            ax.plot(df.index, df['Close'], label='Close Price', linewidth=2, color='black')
            ax.plot(df.index, df['BB_Upper'], label='Upper Band', linewidth=1.5, linestyle='--', color='red')
            ax.plot(df.index, df['BB_Middle'], label='Middle Band (MA20)', linewidth=1.5, linestyle='--', color='blue')
            ax.plot(df.index, df['BB_Lower'], label='Lower Band', linewidth=1.5, linestyle='--', color='green')
            ax.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1)
            ax.set_title(f"{ticker} - {title}")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_moving_averages_comparison(data_dict, title="Moving Averages (20, 50, 200-day)"):
    """Plot multiple moving averages for comparison"""
    tickers = list(data_dict.keys())
    n_tickers = len(tickers)
    
    fig, axes = plt.subplots(n_tickers, 1, figsize=(12, 4*n_tickers))
    if n_tickers == 1:
        axes = [axes]
    
    for idx, (ticker, df) in enumerate(data_dict.items()):
        ax = axes[idx]
        ax.plot(df.index, df['Close'], label='Close Price', linewidth=2, alpha=0.7)
        if 'MA20' in df.columns:
            ax.plot(df.index, df['MA20'], label='20-Day MA', linewidth=1.5)
        if 'MA50' in df.columns:
            ax.plot(df.index, df['MA50'], label='50-Day MA', linewidth=1.5)
        if 'MA200' in df.columns:
            ax.plot(df.index, df['MA200'], label='200-Day MA', linewidth=1.5)
        
        ax.set_title(f"{ticker} - {title}")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def display_model_stats(stats):
    """Display model performance statistics in a Streamlit dataframe"""
    stats = {
        'Model': list(stats.keys()),
        'RMSE': [s['RMSE'] for s in stats.values()],
        'MAPE': [s['MAPE'] for s in stats.values()],
        'R²': [s['R²'] for s in stats.values()],
    }   
    
    stats_df = pd.DataFrame(stats).T
    st.dataframe(stats_df, use_container_width=True)
