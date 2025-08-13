import pandas as pd

def analyze_bollinger_bands(df: pd.DataFrame) -> str:
    """
    Analyzes the latest Bollinger Bands data and returns a text interpretation.
    """
    if df.empty or not all(k in df.columns for k in ['Close', 'BB_High', 'BB_Low']):
        return "Bollinger Bands data is not available for analysis."
    
    latest = df.iloc[-1]
    close = latest['Close']
    bb_high = latest['BB_High']
    bb_low = latest['BB_Low']
    
    analysis = f"The bands are currently between ${bb_low:,.2f} and ${bb_high:,.2f}. "
    
    if close >= bb_high:
        analysis += f"The current price of **${close:,.2f} is touching or above the upper band**, suggesting the asset may be **overbought** and could be due for a short-term price correction."
    elif close <= bb_low:
        analysis += f"The current price of **${close:,.2f} is touching or below the lower band**, suggesting the asset may be **oversold** and could be poised for a rebound."
    else:
        analysis += f"The current price of **${close:,.2f} is trading within the bands**, which does not indicate an immediate overbought or oversold condition."
        
    return analysis

def analyze_rsi(df: pd.DataFrame) -> str:
    """
    Analyzes the latest RSI data and returns a text interpretation.
    """
    if df.empty or 'RSI' not in df.columns:
        return "RSI data is not available for analysis."
        
    latest_rsi = df['RSI'].iloc[-1]
    
    analysis = f"The current RSI is **{latest_rsi:.2f}**. "
    
    if latest_rsi >= 70:
        analysis += "A value **above 70** indicates the asset is likely **overbought**, which can signal a potential price pullback."
    elif latest_rsi <= 30:
        analysis += "A value **below 30** indicates the asset is likely **oversold**, which can signal a potential price rebound."
    else:
        analysis += "This value is in the **neutral zone**, suggesting the market is not currently in an extreme overbought or oversold condition."
        
    return analysis

# We will add more functions here for MACD, Stochastics, etc. in the future.