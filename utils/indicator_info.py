"""
Technical Indicators Information and Explanations
Used in Stock Market Analysis
"""

INDICATORS_INFO = {
    "Moving Averages (MA)": {
        "description": "Moving averages smooth out price data to identify trends.",
        "types": {
            "MA20": "20-day moving average - Short-term trend indicator",
            "MA50": "50-day moving average - Medium-term trend indicator",
            "MA200": "200-day moving average - Long-term trend indicator"
        },
        "interpretation": {
            "Bullish Signal": "Price above MA20 above MA50 above MA200",
            "Bearish Signal": "Price below MA20 below MA50 below MA200",
            "Support/Resistance": "Price tends to bounce off moving average lines"
        },
        "use_case": "Identify trend direction, support/resistance levels, and crossover signals"
    },
    
    "RSI (Relative Strength Index)": {
        "description": "Momentum oscillator that measures the magnitude of price changes to evaluate overbought/oversold conditions.",
        "range": "0 to 100",
        "thresholds": {
            "Overbought": "RSI > 70 (potential sell signal or correction)",
            "Oversold": "RSI < 30 (potential buy signal or bounce)",
            "Neutral": "RSI between 30-70"
        },
        "interpretation": {
            "Strong Uptrend": "RSI consistently above 50",
            "Strong Downtrend": "RSI consistently below 50",
            "Divergence": "Price makes new high but RSI doesn't - potential reversal signal"
        },
        "use_case": "Identify overbought/oversold conditions, potential trend reversals, and momentum strength",
        "calculation": "RSI = 100 - (100 / (1 + RS)) where RS = Avg Gain / Avg Loss"
    },
    
    "MACD (Moving Average Convergence Divergence)": {
        "description": "Trend-following momentum indicator that shows relationship between two moving averages.",
        "components": {
            "MACD Line": "12-day EMA minus 26-day EMA",
            "Signal Line": "9-day EMA of MACD line",
            "Histogram": "Difference between MACD and Signal Line (shows momentum)"
        },
        "interpretation": {
            "Bullish Signal": "MACD crosses above Signal Line (Golden Cross)",
            "Bearish Signal": "MACD crosses below Signal Line (Death Cross)",
            "Momentum": "Larger histogram bars indicate stronger momentum",
            "Divergence": "Price makes new high but MACD doesn't - potential reversal"
        },
        "use_case": "Identify trend changes, momentum shifts, and entry/exit points",
        "best_for": "Trending markets (less effective in sideways markets)"
    },
    
    "Bollinger Bands": {
        "description": "Volatility bands placed above and below a moving average. Shows overbought/oversold levels based on volatility.",
        "components": {
            "Upper Band": "MA20 + (2 × Standard Deviation)",
            "Middle Band": "20-day Simple Moving Average",
            "Lower Band": "MA20 - (2 × Standard Deviation)"
        },
        "interpretation": {
            "Band Width": "Wide bands = high volatility, Narrow bands = low volatility (may precede breakout)",
            "Price at Upper Band": "Potential overbought condition or strong uptrend",
            "Price at Lower Band": "Potential oversold condition or strong downtrend",
            "Squeeze": "Bands narrow - expect price breakout soon"
        },
        "use_case": "Identify volatility levels, overbought/oversold conditions, and potential breakouts",
        "default_settings": "20-period MA with 2 standard deviations"
    },
    
    "ATR (Average True Range)": {
        "description": "Measures market volatility by analyzing the range of price movement. Higher ATR = higher volatility.",
        "use_case": "Determine volatility level, set stop-loss levels, identify breakout opportunities",
        "interpretation": {
            "High ATR": "Market is volatile - larger price swings, potentially good for traders",
            "Low ATR": "Market is quiet - smaller price swings, consolidation phase",
            "ATR Increase": "Volatility increasing - may signal trend change or acceleration",
            "ATR Decrease": "Volatility decreasing - market consolidating"
        },
        "practical_use": "Stop-loss placement = Entry price ± (2 × ATR)",
        "calculation": "ATR = 14-period average of True Range (max of: High-Low, |High-Close|, |Low-Close|)"
    },
    
    "Volatility": {
        "description": "Rolling 20-day standard deviation of daily returns. Measures price fluctuation intensity.",
        "interpretation": {
            "High Volatility": "> 3-4% - High risk, high potential reward",
            "Medium Volatility": "2-3% - Normal market conditions",
            "Low Volatility": "< 2% - Stable prices, potential consolidation"
        },
        "use_case": "Compare risk levels between stocks, identify consolidation periods, portfolio risk assessment",
        "investor_implications": {
            "Risk-Averse": "Prefer low volatility stocks for stability",
            "Traders": "Prefer high volatility for trading opportunities"
        }
    },
    
    "Volume": {
        "description": "Number of shares traded. Higher volume confirms price movements and trend strength.",
        "interpretation": {
            "High Volume": "Strong conviction behind the price move",
            "Low Volume": "Weak move that may reverse",
            "Volume Spike": "Significant event or strong breakout",
            "Green Candle": "Close > Open - Bullish volume (buying pressure)",
            "Red Candle": "Close < Open - Bearish volume (selling pressure)"
        },
        "use_case": "Confirm price trends, identify breakouts, assess trend strength",
        "rule": "Price move on high volume is more reliable than price move on low volume"
    },
    
    "Daily Returns": {
        "description": "Percentage change in closing price from one day to the next.",
        "calculation": "(Today's Close - Yesterday's Close) / Yesterday's Close × 100%",
        "use_case": "Measure daily performance, calculate volatility, identify winning/losing days",
        "interpretation": {
            "Positive Return": "Stock gained value that day",
            "Negative Return": "Stock lost value that day",
            "Large Return": "Significant event or catalyst affecting the stock"
        }
    }
}

def get_indicator_description(indicator_name):
    """Get detailed description of an indicator"""
    return INDICATORS_INFO.get(indicator_name, "Indicator not found")

def display_all_indicators_info():
    """Return formatted text with all indicators information"""
    text = "# 📊 Technical Indicators Guide\n\n"
    
    for indicator, info in INDICATORS_INFO.items():
        text += f"## {indicator}\n"
        text += f"**Description:** {info.get('description', 'N/A')}\n\n"
        
        if 'range' in info:
            text += f"**Range:** {info['range']}\n\n"
        
        if 'calculation' in info:
            text += f"**Calculation:** {info['calculation']}\n\n"
        
        if 'components' in info:
            text += "**Components:**\n"
            for comp, desc in info['components'].items():
                text += f"- {comp}: {desc}\n"
            text += "\n"
        
        if 'types' in info:
            text += "**Types:**\n"
            for typ, desc in info['types'].items():
                text += f"- {typ}: {desc}\n"
            text += "\n"
        
        if 'thresholds' in info:
            text += "**Key Thresholds:**\n"
            for threshold, desc in info['thresholds'].items():
                text += f"- {threshold}: {desc}\n"
            text += "\n"
        
        if 'interpretation' in info:
            text += "**Interpretation:**\n"
            for signal, desc in info['interpretation'].items():
                text += f"- {signal}: {desc}\n"
            text += "\n"
        
        if 'use_case' in info:
            text += f"**Use Case:** {info['use_case']}\n\n"
        
        if 'practical_use' in info:
            text += f"**Practical Use:** {info['practical_use']}\n\n"
        
        text += "---\n\n"
    
    return text

def get_quick_reference():
    """Quick reference guide for indicator signals"""
    signals = {
        "Bullish Signals": [
            "• Price above MA20 > MA50 > MA200",
            "• RSI > 50 and increasing",
            "• MACD above Signal Line and histogram positive",
            "• Price bouncing off Bollinger Lower Band",
            "• High volume on green candles",
            "• Rising volatility (potential breakout)"
        ],
        "Bearish Signals": [
            "• Price below MA20 < MA50 < MA200",
            "• RSI < 50 and decreasing",
            "• MACD below Signal Line and histogram negative",
            "• Price near Bollinger Upper Band",
            "• High volume on red candles",
            "• Decreasing volatility (consolidation)"
        ],
        "Overbought/Oversold": [
            "• RSI > 70 = Overbought (potential sell)",
            "• RSI < 30 = Oversold (potential buy)",
            "• Price near Bollinger Upper Band = Overbought",
            "• Price near Bollinger Lower Band = Oversold"
        ]
    }
    
    text = "# 📌 Quick Reference - Indicator Signals\n\n"
    for category, signals_list in signals.items():
        text += f"## {category}\n"
        for signal in signals_list:
            text += f"{signal}\n"
        text += "\n"
    
    return text
