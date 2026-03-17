import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from services.data_loader import load_data
from models.lstm_model import lstm_predict
from visuals import plots
from utils.indicator_info import display_all_indicators_info, get_quick_reference
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock LSTM + RL Predictor", layout="wide")

# Popular stocks list
popular_stocks = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM",
    "V", "JNJ", "WMT", "PG", "MA", "NFLX", "ADBE", "RELIANCE.BSE",
    "TCS.BSE", "INFY.BSE", "HDFC.BSE", "ICICIBANK.BSE", "SBIN.BSE"
]

# Create navigation tabs
page = st.sidebar.radio("Navigation", ["📈 Analysis", "🔄 Compare Stocks", "📚 Indicators Info"])

st.title("📈 Stock Market Prediction & Recommendation")


def process_stock(ticker, start_date, end_date, price_type, forecast_days):
    """Process and analyze a single stock"""
    try:
        st.info(f"📥 Loading {ticker} data from {start_date} to {end_date}...")
        df = load_data(ticker, start_date, end_date)
        
        if df.empty:
            st.error(f"❌ No data found for {ticker}. Check the ticker symbol.")
        else:
            st.success(f"✅ Loaded {len(df)} days of data")
            
            # Show price chart
            st.subheader("📊 Price Chart")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index, df[price_type], label=price_type, linewidth=2)
            ax.set_title(f"{ticker} {price_type} Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            
            # Train model
            st.info(f"🤖 Training ensemble model...")
            actual, pred, forecast, rmse, mape = lstm_predict(df, price_type, forecast_days)
            st.success("✅ Model trained successfully!")
            
            # Show metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE", f"${rmse:.2f}")
            with col2:
                st.metric("MAPE", f"{mape:.2f}%")
            
            # Show results table
            st.subheader("📈 Prediction Results")
            
            # Historical predictions
            results_df = pd.DataFrame({
                "Actual": actual[-20:],
                "Predicted": pred[-20:],
                "Error ($)": actual[-20:] - pred[-20:]
            })
            st.dataframe(results_df, use_container_width=True)
            
            # Forecast
            st.subheader("🔮 Price Forecast")
            future_dates = pd.date_range(df.index[-1], periods=forecast_days+1, freq='D')[1:]
            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Forecast": forecast
            })
            st.dataframe(forecast_df, use_container_width=True)
            
            # Forecast chart - ensure all arrays have matching lengths
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Use only the last 60 actual and predicted (matching lengths)
            lookback = min(60, len(actual))
            actual_plot = actual[-lookback:]
            pred_plot = pred[-lookback:]
            dates_plot = df.index[-lookback:]
            
            ax.plot(dates_plot, actual_plot, label="Actual", linewidth=2, marker='o', markersize=3)
            ax.plot(dates_plot, pred_plot, label="Predicted", linewidth=2, alpha=0.7, marker='s', markersize=3)
            ax.plot(future_dates, forecast, label="Forecast", linewidth=2, linestyle='--', color='red', marker='^', markersize=4)
            ax.set_title(f"{ticker} - Actual vs Predicted vs Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
            
            # Trading recommendation
            st.subheader("🎯 Trading Recommendation")
            current_price = actual[-1]
            forecast_price = forecast[-1]
            return_pct = ((forecast_price - current_price) / current_price) * 100
            
            if return_pct >= 2:
                st.success(f"✅ **BUY** - Expected return: {return_pct:.2f}%")
            elif return_pct <= -2:
                st.error(f"❌ **SELL** - Expected return: {return_pct:.2f}%")
            else:
                st.info(f"⏸️ **HOLD** - Expected return: {return_pct:.2f}%")
            
            st.write(f"Current: ${current_price:.2f} → Forecast: ${forecast_price:.2f}")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# ============= INDICATORS INFO PAGE =============
if page == "📚 Indicators Info":
    st.header("📚 Technical Indicators Guide")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📖 Full Indicators Guide"):
            st.write(display_all_indicators_info())
    with col2:
        if st.button("⚡ Quick Reference"):
            st.write(get_quick_reference())
    
    st.markdown("---")
    st.subheader("📊 Popular Technical Indicators")
    
    # Create tabs for each indicator
    ind_tabs = st.tabs(["Moving Averages", "RSI", "MACD", "Bollinger Bands", "ATR", "Volatility"])
    
    with ind_tabs[0]:
        st.markdown("""
        ## Moving Averages (MA)
        **Purpose:** Smooth price data to identify trends and support/resistance levels
        
        - **MA20**: 20-day moving average (short-term trend)
        - **MA50**: 50-day moving average (medium-term trend)
        - **MA200**: 200-day moving average (long-term trend)
        
        ### Signals:
        - **Bullish**: Price > MA20 > MA50 > MA200
        - **Bearish**: Price < MA20 < MA50 < MA200
        - **Golden Cross**: MA50 crosses above MA200 (bullish)
        - **Death Cross**: MA50 crosses below MA200 (bearish)
        """)
    
    with ind_tabs[1]:
        st.markdown("""
        ## RSI (Relative Strength Index)
        **Purpose:** Identify overbought/oversold conditions and momentum strength
        
        **Range:** 0-100
        
        ### Interpretation:
        - **RSI > 70**: Overbought (potential sell signal)
        - **RSI < 30**: Oversold (potential buy signal)
        - **RSI 50+**: Strong uptrend
        - **RSI 50-**: Strong downtrend
        
        ### Trading Rules:
        - Overbought doesn't mean sell immediately - strong trends can stay overbought
        - Look for divergence: Price makes new high but RSI doesn't
        """)
    
    with ind_tabs[2]:
        st.markdown("""
        ## MACD (Moving Average Convergence Divergence)
        **Purpose:** Identify trend changes and momentum shifts
        
        **Components:**
        - **MACD Line**: 12-day EMA - 26-day EMA
        - **Signal Line**: 9-day EMA of MACD
        - **Histogram**: MACD - Signal Line (shows momentum)
        
        ### Signals:
        - **Bullish**: MACD crosses above Signal Line
        - **Bearish**: MACD crosses below Signal Line
        - **Momentum**: Larger histogram = stronger trend
        """)
    
    with ind_tabs[3]:
        st.markdown("""
        ## Bollinger Bands
        **Purpose:** Identify volatility and overbought/oversold levels
        
        **Components:**
        - **Upper Band**: MA20 + (2 × Std Dev)
        - **Middle Band**: 20-day MA
        - **Lower Band**: MA20 - (2 × Std Dev)
        
        ### Interpretation:
        - **Narrow Bands**: Low volatility (potential breakout)
        - **Wide Bands**: High volatility
        - **Price at Upper Band**: Overbought/Strong uptrend
        - **Price at Lower Band**: Oversold/Strong downtrend
        """)
    
    with ind_tabs[4]:
        st.markdown("""
        ## ATR (Average True Range)
        **Purpose:** Measure market volatility and set stop-loss levels
        
        **Interpretation:**
        - **High ATR**: Market is volatile (larger price swings)
        - **Low ATR**: Market is quiet (consolidation)
        - **Rising ATR**: Volatility increasing
        - **Falling ATR**: Volatility decreasing
        
        ### Practical Use:
        Stop-Loss Placement = Entry Price ± (2 × ATR)
        """)
    
    with ind_tabs[5]:
        st.markdown("""
        ## Volatility (Rolling)
        **Purpose:** Measure price fluctuation intensity
        
        **Calculation:** 20-day standard deviation of daily returns
        
        ### Interpretation:
        - **High Volatility (>3-4%)**: High risk, high reward potential
        - **Medium Volatility (2-3%)**: Normal conditions
        - **Low Volatility (<2%)**: Stable, potential consolidation
        
        ### Use Cases:
        - Compare risk levels between stocks
        - Identify consolidation periods
        - Portfolio risk assessment
        """)

# ============= COMPARISON PAGE =============
elif page == "🔄 Compare Stocks":
    st.header("🔄 Compare Multiple Stocks Side-by-Side")
    
    st.sidebar.header("Comparison Settings")
    
    # Number of stocks to compare
    num_compare = st.sidebar.number_input("How many stocks to compare?", min_value=2, max_value=5, value=2, key="compare_num")
    
    # Select stocks
    compare_tickers = []
    for i in range(num_compare):
        ticker = st.sidebar.selectbox(
            f"Stock {i+1}",
            popular_stocks,
            key=f"compare_ticker_{i}"
        )
        if ticker:
            compare_tickers.append(ticker)
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        compare_start = st.date_input("Start Date", value=None, key="compare_start_date")
    with col2:
        compare_end = st.date_input("End Date", value=None, key="compare_end_date")
    
    # Run comparison
    if st.sidebar.button("🔄 Compare", key="compare_btn"):
        if len(compare_tickers) < 2:
            st.error("❌ Please select at least 2 stocks")
        elif compare_start is None or compare_end is None:
            st.error("❌ Please select both start and end dates")
        elif compare_start >= compare_end:
            st.error("❌ Start date must be before end date")
        else:
            # Load all data
            data_dict = {}
            st.info("📥 Loading data for all stocks...")
            
            for ticker in compare_tickers:
                try:
                    df = load_data(ticker, compare_start, compare_end)
                    if not df.empty:
                        data_dict[ticker] = df
                    else:
                        st.warning(f"⚠️ No data found for {ticker}")
                except Exception as e:
                    st.warning(f"⚠️ Error loading {ticker}: {str(e)}")
            
            if data_dict:
                st.success(f"✅ Loaded data for {len(data_dict)} stocks")
                
                # Create comparison tabs
                comp_tabs = st.tabs([
                    "📊 Price Comparison",
                    "💹 Performance",
                    "📈 Normalized Prices",
                    "📉 Volatility",
                    "💪 RSI",
                    "📊 MACD",
                    "📐 Bollinger Bands",
                    "🎯 Moving Averages"
                ])
                
                with comp_tabs[0]:
                    st.subheader("Stock Price Comparison")
                    plots.plot_compare(data_dict)
                
                with comp_tabs[1]:
                    st.subheader("Performance Comparison (%)")
                    plots.plot_performance_comparison(data_dict)
                
                with comp_tabs[2]:
                    st.subheader("Normalized Price Comparison (0-1 scale)")
                    plots.plot_normalized_comparison(data_dict)
                
                with comp_tabs[3]:
                    st.subheader("Volatility Comparison")
                    plots.plot_volatility_comparison(data_dict)
                
                with comp_tabs[4]:
                    st.subheader("RSI Comparison")
                    plots.plot_rsi_comparison(data_dict)
                
                with comp_tabs[5]:
                    st.subheader("MACD Comparison")
                    plots.plot_macd_comparison(data_dict)
                
                with comp_tabs[6]:
                    st.subheader("Bollinger Bands Comparison")
                    plots.plot_bollinger_bands_comparison(data_dict)
                
                with comp_tabs[7]:
                    st.subheader("Moving Averages Comparison")
                    plots.plot_moving_averages_comparison(data_dict)

# ============= ANALYSIS PAGE =============
else:
    st.sidebar.header("Settings")
    
    # Number of stocks to analyze
    num_stocks = st.sidebar.number_input("How many stocks to analyze?", min_value=1, max_value=5, value=1)
    
    # Stock selection dropdowns
    selected_tickers = []
    for i in range(num_stocks):
        ticker = st.sidebar.selectbox(
            f"Stock {i+1}",
            popular_stocks,
            key=f"ticker_{i}"
        )
        if ticker:
            selected_tickers.append(ticker)

    price_type = st.sidebar.radio("Price Type", ["Close", "Open"], index=0)

    # Date inputs - explicitly set to None to clear cache
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=None, key="start_date_input")
    with col2:
        end_date = st.date_input("End Date", value=None, key="end_date_input")

    forecast_days = st.sidebar.slider("Forecast Days", 1, 60, 30)

    # Run button
    if st.sidebar.button("🚀 Run Analysis"):
        if not selected_tickers:
            st.error("❌ Please select at least one stock ticker")
        elif start_date is None or end_date is None:
            st.error("❌ Please select both start and end dates")
        elif start_date >= end_date:
            st.error("❌ Start date must be before end date")
        else:
            # Create tabs for each stock if multiple
            if len(selected_tickers) > 1:
                tabs = st.tabs([f"📊 {t}" for t in selected_tickers])
            else:
                tabs = [None]
            
            for idx, ticker in enumerate(selected_tickers):
                # Use tab if multiple stocks, otherwise use main st
                if len(selected_tickers) > 1:
                    container = tabs[idx]
                    is_tab = True
                else:
                    container = st
                    is_tab = False
                
                try:
                    if is_tab:
                        with tabs[idx]:
                            st.subheader(f"Analyzing {ticker}")
                            process_stock(ticker, start_date, end_date, price_type, forecast_days)
                    else:
                        process_stock(ticker, start_date, end_date, price_type, forecast_days)
                
                except Exception as e:
                    st.error(f"❌ Error analyzing {ticker}: {str(e)}")
                    st.code(str(e))
