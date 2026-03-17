import yfinance as yf
import streamlit as st
from pipeline.etl import clean_stock_data, add_features


@st.cache_data(ttl=21600)  # 6-hour cache
def load_data(ticker, start, end):
    """Fetch stock data from Yahoo Finance and apply ETL."""
    print(f"📥 Loading {ticker} from {start} to {end}...")
    
    # Request extra days to work around yfinance's 250-day business day limit
    # yfinance seems to have a hard cap around 250 business days per request
    # So we request 1.5x the intended range and trim to actual range
    from datetime import timedelta
    expanded_start = start - timedelta(days=int((end - start).days * 0.3))
    expanded_end = end + timedelta(days=int((end - start).days * 0.3))
    
    df = yf.download(str(ticker), start=expanded_start, end=expanded_end, progress=False)
    
    if df.empty:
        print(f"❌ No data found for {ticker}")
        return df
    
    # Trim to requested date range
    df = df.loc[(df.index.date >= start) & (df.index.date <= end)]
    
    print(f"📊 Downloaded {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
    expected_days = (end - start).days
    if len(df) < expected_days * 0.5:
        print(f"⚠️  WARNING: Got {len(df)} days, expected ~{expected_days}")
    
    df = clean_stock_data(df)
    df = add_features(df)
    return df
