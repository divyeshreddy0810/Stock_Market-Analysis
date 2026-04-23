import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import importlib.util
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from services.data_loader import load_data
from models.random_forest_model import random_forest_predict
from visuals import plots
from utils.indicator_info import display_all_indicators_info, get_quick_reference
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Random Forest Predictor", layout="wide")

# Popular stocks list
popular_stocks = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM",
    "V", "JNJ", "WMT", "PG", "MA", "NFLX", "ADBE", "RELIANCE.BSE",
    "TCS.BSE", "INFY.BSE", "HDFC.BSE", "ICICIBANK.BSE", "SBIN.BSE"
]

# Create navigation tabs
page = st.sidebar.radio("Navigation", ["📈 Analysis", "🔄 Compare Stocks", "📚 Indicators Info", "🧪 ML Model Results"])

st.title("📈 Stock Market Prediction & Recommendation")

TRAINING_WINDOW_YEARS = 3


def get_training_start_date(end_date, years=TRAINING_WINDOW_YEARS):
    """Return training start date for a rolling N-year window ending at end_date."""
    return end_date - datetime.timedelta(days=365 * years)


def load_metrics_evaluator():
    """Load metrics module from path containing spaces."""
    metrics_path = Path(__file__).parent / "Results for ML_models" / "metrics_evaluator.py"
    spec = importlib.util.spec_from_file_location("metrics_evaluator", str(metrics_path))
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load metrics evaluator module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_supervised_dataset(prices, seq_len=10):
    """Create sliding-window features and targets for classic ML models."""
    if seq_len < 1:
        raise ValueError("Sequence length must be at least 1")
    if len(prices) <= seq_len:
        raise ValueError(f"Need more than {seq_len} points, got {len(prices)}")

    X, y = [], []
    for i in range(len(prices) - seq_len):
        X.append(prices[i:i + seq_len])
        y.append(prices[i + seq_len])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError(f"Invalid feature matrix shape: {X.shape}")

    return X, y


def sanitize_prices(prices):
    """Replace NaNs in a price array with the series mean."""
    prices = np.array(prices, dtype=float)
    valid = prices[~np.isnan(prices)]
    if len(valid) == 0:
        raise ValueError("Price series has no valid values")
    return np.nan_to_num(prices, nan=np.nanmean(valid))


def build_multiticker_training_dataset(training_frames, price_type, seq_len=10):
    """Build one normalized training set by concatenating windows from multiple tickers."""
    X_parts, y_parts = [], []
    used_tickers = []

    for ticker, df in training_frames.items():
        if price_type not in df.columns:
            continue

        prices = sanitize_prices(df[price_type].values)
        if len(prices) <= seq_len:
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        X_ticker, y_ticker = build_supervised_dataset(prices_scaled, seq_len=seq_len)

        X_parts.append(X_ticker)
        y_parts.append(y_ticker)
        used_tickers.append(ticker)

    if not X_parts:
        raise ValueError("No valid training windows were created from selected datasets")

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    return X_train, y_train, used_tickers


def run_multiticker_model(
    model,
    training_frames,
    target_df,
    price_type,
    forecast_days,
    seq_len,
    evaluate_model_performance,
):
    """Train on multiple ticker datasets and evaluate/forecast on the target ticker."""
    X_train, y_train, used_tickers = build_multiticker_training_dataset(
        training_frames,
        price_type,
        seq_len=seq_len,
    )

    if price_type not in target_df.columns:
        raise ValueError(f"Column '{price_type}' not found in target dataset")

    target_prices = sanitize_prices(target_df[price_type].values)
    if len(target_prices) <= seq_len:
        raise ValueError(f"Need more than {seq_len} target points, got {len(target_prices)}")

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaled = target_scaler.fit_transform(target_prices.reshape(-1, 1)).flatten()
    X_eval, y_eval = build_supervised_dataset(target_scaled, seq_len=seq_len)

    model.fit(X_train, y_train)

    y_pred_eval_scaled = model.predict(X_eval)
    y_true_eval = target_scaler.inverse_transform(y_eval.reshape(-1, 1)).flatten()
    y_pred_eval = target_scaler.inverse_transform(y_pred_eval_scaled.reshape(-1, 1)).flatten()
    metrics = evaluate_model_performance(y_true_eval, y_pred_eval)

    last_sequence = target_scaled[-seq_len:].copy()
    forecast_scaled = []
    for _ in range(forecast_days):
        next_scaled = model.predict(last_sequence.reshape(1, -1))[0]
        forecast_scaled.append(next_scaled)
        last_sequence = np.append(last_sequence[1:], next_scaled)

    forecast = target_scaler.inverse_transform(
        np.array(forecast_scaled).reshape(-1, 1)
    ).flatten()

    return {
        "predictions": y_pred_eval,
        "actual": y_true_eval,
        "metrics": metrics,
        "forecast": forecast,
        "used_tickers": used_tickers,
        "train_samples": len(X_train),
    }


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
            
            # Train model on past N years ending at selected end date.
            training_start = get_training_start_date(end_date)
            st.info(
                f"🤖 Training model on past {TRAINING_WINDOW_YEARS} years "
                f"({training_start} to {end_date})..."
            )
            training_df = load_data(ticker, training_start, end_date)
            if training_df.empty:
                st.error("❌ Unable to load enough training data for model training")
                return

            actual, pred, forecast, rmse, mape = random_forest_predict(
                training_df,
                price_type,
                forecast_days,
                evaluation_df=df,
            )
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
elif page == "🧪 ML Model Results":
    st.header("🧪 ML Model Results")
    st.write("Compare ML model performance using RMSE, R2, RSS, MAPE, F-Measure, and Confusion Matrix.")

    st.sidebar.header("ML Results Settings")

    ml_ticker = st.sidebar.selectbox("Stock", popular_stocks, key="ml_ticker")
    ml_price_type = st.sidebar.radio("Price Type", ["Close", "Open"], index=0, key="ml_price_type")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        ml_start = st.date_input("Start Date", value=None, key="ml_start_date")
    with col2:
        ml_end = st.date_input("End Date", value=None, key="ml_end_date")

    st.sidebar.caption(
        f"Model training uses past {TRAINING_WINDOW_YEARS} years ending at selected End Date."
    )

    ml_forecast_days = st.sidebar.slider("Forecast Days", 5, 60, 30, key="ml_forecast_days")

    default_training_tickers = [ml_ticker]
    for candidate in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]:
        if candidate not in default_training_tickers:
            default_training_tickers.append(candidate)
        if len(default_training_tickers) == 3:
            break

    training_tickers = st.sidebar.multiselect(
        "Training Datasets (3+ yfinance tickers)",
        popular_stocks,
        default=default_training_tickers,
        key="ml_training_tickers",
        help="Select at least 3 ticker datasets to train the model, then evaluate on the selected stock.",
    )

    selected_models = st.sidebar.multiselect(
        "Select Models",
        ["Random Forest", "Linear Regression", "SVM"],
        default=["Random Forest", "Linear Regression", "SVM"],
        key="ml_selected_models"
    )

    if st.sidebar.button("📊 Run ML Results", key="ml_results_btn"):
        if not selected_models:
            st.error("❌ Please select at least one ML model")
        elif len(training_tickers) < 3:
            st.error("❌ Please select at least 3 training datasets")
        elif ml_start is None or ml_end is None:
            st.error("❌ Please select both start and end dates")
        elif ml_start >= ml_end:
            st.error("❌ Start date must be before end date")
        else:
            try:
                ml_train_start = get_training_start_date(ml_end)
                st.info(
                    f"📥 Loading target data ({ml_start} to {ml_end}) and training window "
                    f"({ml_train_start} to {ml_end}) from yfinance..."
                )
                target_df = load_data(ml_ticker, ml_start, ml_end)

                if target_df.empty:
                    st.error("❌ No data found for the selected prediction stock/date range")
                else:
                    training_frames = {}
                    for train_ticker in training_tickers:
                        train_df = load_data(train_ticker, ml_train_start, ml_end)
                        if train_df.empty:
                            st.warning(f"⚠️ No data found for training dataset: {train_ticker}")
                            continue
                        training_frames[train_ticker] = train_df

                    if len(training_frames) < 3:
                        st.error("❌ Could not load at least 3 training datasets. Adjust tickers/date range and retry.")
                    else:
                        metrics_module = load_metrics_evaluator()
                        evaluate_model_performance = metrics_module.evaluate_model_performance
                        seq_len = 10

                        model_builders = {
                            "Random Forest": lambda: RandomForestRegressor(n_estimators=100, random_state=42),
                            "Linear Regression": LinearRegression,
                            "SVM": lambda: SVR(kernel='rbf', C=1e3, gamma=0.1),
                        }

                        summary_rows = []
                        model_outputs = {}

                        st.success(
                            f"✅ Loaded {len(training_frames)} training datasets. "
                            f"Predicting {ml_ticker} using target range {ml_start} to {ml_end}."
                        )

                        for model_name in selected_models:
                            try:
                                model = model_builders[model_name]()
                                output = run_multiticker_model(
                                    model=model,
                                    training_frames=training_frames,
                                    target_df=target_df,
                                    price_type=ml_price_type,
                                    forecast_days=ml_forecast_days,
                                    seq_len=seq_len,
                                    evaluate_model_performance=evaluate_model_performance,
                                )

                                metrics = output["metrics"]
                                model_outputs[model_name] = output

                                summary_rows.append({
                                    "Model": model_name,
                                    "Train Datasets": len(output["used_tickers"]),
                                    "Train Samples": output["train_samples"],
                                    "RMSE": metrics["RMSE"],
                                    "R2": metrics["R2_Score"],
                                    "RSS": metrics["RSS"],
                                    "MAPE (%)": metrics["MAPE"] * 100,
                                    "F-Measure": metrics["F_Measure"],
                                })
                            except Exception as model_error:
                                st.warning(f"⚠️ {model_name} failed: {str(model_error)}")
                                continue

                    if not summary_rows:
                        st.error("❌ No model results were generated")
                    else:
                        st.subheader("📋 Model Metrics Summary")
                        summary_df = pd.DataFrame(summary_rows).sort_values("RMSE")
                        st.dataframe(summary_df, use_container_width=True)

                        st.subheader("🔍 Detailed Metrics")
                        for model_name, output in model_outputs.items():
                            with st.expander(f"{model_name} Details", expanded=False):
                                metrics = output["metrics"]

                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                                    st.metric("RSS", f"{metrics['RSS']:.4f}")
                                with c2:
                                    st.metric("R2", f"{metrics['R2_Score']:.4f}")
                                    st.metric("MAPE", f"{metrics['MAPE'] * 100:.2f}%")
                                with c3:
                                    st.metric("F-Measure", f"{metrics['F_Measure']:.4f}")

                                conf_df = pd.DataFrame(
                                    metrics["Confusion_Matrix"],
                                    index=["Actual Down", "Actual Up"],
                                    columns=["Pred Down", "Pred Up"]
                                )
                                st.write("Confusion Matrix (Direction of price movement)")
                                st.dataframe(conf_df, use_container_width=True)

                                st.write(
                                    f"Training datasets used: {', '.join(output['used_tickers'])} "
                                    f"({output['train_samples']} windows)"
                                )

                                lookback = len(output["predictions"])
                                pred_dates = target_df.index[seq_len:seq_len + lookback]

                                fig, ax = plt.subplots(figsize=(12, 5))
                                ax.plot(pred_dates, output["actual"], label="Actual", linewidth=2)
                                ax.plot(pred_dates, output["predictions"], label="Predicted", linewidth=2, linestyle="--")
                                ax.set_title(f"{model_name}: Actual vs Predicted ({ml_ticker})")
                                ax.set_xlabel("Date")
                                ax.set_ylabel("Price ($)")
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig)

                                if len(output["forecast"]) > 0:
                                    st.write("Forecast Preview")
                                    future_dates = pd.date_range(target_df.index[-1], periods=len(output["forecast"]) + 1, freq='D')[1:]
                                    forecast_df = pd.DataFrame({
                                        "Date": future_dates,
                                        "Forecast": output["forecast"]
                                    })
                                    st.dataframe(forecast_df.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error generating ML results: {str(e)}")

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

    # Date inputs
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=None, key="start_date_input")
    with col2:
        end_date = st.date_input("End Date", value=None, key="end_date_input")

    st.sidebar.caption(
        f"Model training uses past {TRAINING_WINDOW_YEARS} years ending at selected End Date."
    )

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
