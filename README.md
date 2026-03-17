# Stock Market Prediction and Recommendation

Interactive Streamlit application for stock analysis, short-term forecasting, and technical-indicator-based comparison across multiple stocks.

## Features
- Single-stock analysis with historical price visualization
- Multi-stock comparison dashboard (price, performance, volatility, RSI, MACD, Bollinger Bands, moving averages)
- ML-based forecasting ensemble (Gradient Boosting, Random Forest, SVR)
- Forecast horizon selection (1 to 60 days)
- Error metrics: RMSE and MAPE
- Technical indicators ETL pipeline (MA20/50/200, RSI, MACD, ATR, Bollinger Bands, volatility)
- Built-in indicator guide and quick reference

## Project Structure
```
Stock_Project/
|-- app.py
|-- requirements.txt
|-- .streamlit/config.toml
|-- models/
|   |-- lstm_model.py
|   `-- rl_agent.py
|-- pipeline/
|   `-- etl.py
|-- services/
|   `-- data_loader.py
|-- utils/
|   |-- preprocessing.py
|   `-- indicator_info.py
`-- visuals/
	`-- plots.py
```

## Prerequisites
- Python 3.10+
- pip

## Setup and Run
1. Clone the repository and open the project directory.
2. Create a virtual environment.
3. Activate the virtual environment.
4. Install dependencies.
5. Start Streamlit.

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows (PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## How to Use
1. Open the Streamlit URL shown in terminal (usually http://localhost:8501).
2. In the sidebar, choose one of:
   - Analysis: Analyze one or more selected stocks.
   - Compare Stocks: Compare up to 5 stocks side-by-side.
   - Indicators Info: Read technical indicator explanations.
3. Select date range and run analysis/compare.

## Notes
- Data source: Yahoo Finance via `yfinance`.
- Some tickers are exchange-specific (for example `.BSE` suffix for BSE symbols).
- Forecasts are model-driven estimates and should not be considered financial advice.

## Troubleshooting
- If `streamlit` is not found, verify your virtual environment is active.
- If no data loads for a symbol, validate ticker format and selected date range.
- If installation fails, upgrade pip and retry:

```bash
python -m pip install --upgrade pip setuptools wheel
```
