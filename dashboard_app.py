from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import os
import json
import numpy as np
from db_utils import load_forecast_results

# --- Page Configuration ---
st.set_page_config(page_title="305 Crypto Forecast", page_icon="📈", layout="wide")

# --- Path Configuration for local daily data ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

# --- Caching Functions ---
@st.cache_data(ttl=3600)
def get_historical_data():
    """Loads all forecast data from the database."""
    return load_forecast_results()

@st.cache_data(ttl=3600)
def load_chart_data(ticker):
    """Loads the detailed indicator data from the local CSV file."""
    file_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        except Exception:
            return None
    return None

# --- Main Application ---
st.title("📈 305 Crypto Forecast Dashboard")
st.markdown("An automated forecasting and sentiment analysis system for major cryptocurrencies.")

historical_df = get_historical_data()

if historical_df.empty:
    st.error("🚨 No forecast data found in the database. The daily analysis may not have run yet.")
    st.stop()

# --- Data Preparation ---
# Get the most recent forecast date
latest_date = historical_df['Date'].max()
# Filter for only the most recent results to display the "Today's Overview"
latest_forecast_df = historical_df[historical_df['Date'] == latest_date].copy()

# --- Utility Function ---
def format_numeric_columns(df):
    formatted_df = df.copy()
    numeric_cols = [
        'Actual_Price', 'Prophet_Forecast', 'LSTM_Forecast', 'All_Time_High',
        'High', 'Low', 'Close', 'Open', 'Volume', 'SMA', 'EMA',
        'BB_High', 'BB_Low', 'RSI', 'MACD', 'MACD_Signal', 'OBV',
        'Ichimoku_a', 'Ichimoku_b', 'Active_Addresses', 'Transaction_Volume',
        'Forecasted High'
    ]
    for col in numeric_cols:
        if col in formatted_df.columns:
            try:
                # Apply formatting only to numeric types to avoid errors
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x)
            except (ValueError, TypeError):
                pass
    return formatted_df

# --- Sidebar & Main Content ---
st.sidebar.header("Dashboard Options")
selected_coin = st.sidebar.selectbox("Select a Cryptocurrency", latest_forecast_df['Coin'].unique())

chart_data = load_chart_data(selected_coin)
coin_forecast = latest_forecast_df[latest_forecast_df['Coin'] == selected_coin].iloc[0]

# --- Main Page Layout ---
st.header(f"Today's Overview for {selected_coin}")
col1, col2, col3, col4 = st.columns(4)
actual_price = pd.to_numeric(coin_forecast['Actual_Price'], errors='coerce')
all_time_high = pd.to_numeric(coin_forecast['All_Time_High'], errors='coerce')
sentiment_score = pd.to_numeric(coin_forecast['Sentiment_Score'], errors='coerce')
col1.metric("Actual Price", f"${actual_price:,.2f}" if pd.notna(actual_price) else "N/A")
col2.metric("All-Time High", f"${all_time_high:,.2f}" if pd.notna(all_time_high) else "N/A")
col3.metric("Sentiment Score", f"{sentiment_score:.2f}" if pd.notna(sentiment_score) else "N/A")

with st.expander("❓ **Explain the Sentiment Score**"):
    st.info(
        """
        The **Sentiment Score** is an aggregated value from a natural language processing (NLP) analysis of recent news articles.
        - **A positive score** (e.g., > 0.10) indicates a **bullish sentiment** towards the cryptocurrency.
        - **A negative score** (e.g., < -0.10) indicates a **bearish sentiment**.
        - **A score close to zero** indicates a **neutral market sentiment**.
        This score is a key fundamental indicator, as news and public opinion often precede price movements.
        """
    )

st.header("5-Day High Forecast vs. Historical Highs")
if chart_data is not None and 'High_Forecast_5_Day' in coin_forecast and pd.notna(coin_forecast['High_Forecast_5_Day']):
    historical_highs = chart_data[['High']].tail(5)
    historical_highs.index = historical_highs.index.strftime('%Y-%m-%d')
    try:
        forecast_data = json.loads(str(coin_forecast['High_Forecast_5_Day']))
        if forecast_data:
            forecast_df_highs = pd.DataFrame(forecast_data)
            forecast_df_highs['ds'] = pd.to_datetime(forecast_df_highs['ds']).dt.strftime('%Y-%m-%d')
            forecast_df_highs = forecast_df_highs.rename(columns={'ds': 'Date', 'yhat': 'Forecasted High'}).set_index('Date')
            combined_df = pd.concat([historical_highs, forecast_df_highs['Forecasted High']], axis=1)
            st.bar_chart(combined_df)
            st.dataframe(format_numeric_columns(combined_df.reset_index()))
        else:
            st.warning("Forecast data is available but empty.")
    except (json.JSONDecodeError, TypeError) as e:
        st.error(f"Could not parse the 5-day forecast data. Error: {e}")
else:
    st.warning(f"Could not load 5-day forecast data for {selected_coin}.")

st.header(f"Technical Indicators for {selected_coin}")
if chart_data is not None:
    st.subheader("Price, Moving Averages, & Bollinger Bands")
    st.line_chart(chart_data[['Close', 'SMA', 'EMA', 'BB_High', 'BB_Low']])
    st.info(
        """
        **Reasoning:** These indicators help identify the current trend and volatility.
        - **SMA/EMA:** A price above its moving average suggests an uptrend, while a price below suggests a downtrend. Crossovers can signal a change in trend.
        - **Bollinger Bands:** The bands widen during high volatility and narrow during low volatility. A price touching the upper band may suggest it's overbought, while touching the lower band may suggest it's oversold.
        """
    )
    
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.subheader("RSI (Relative Strength Index)")
        st.line_chart(chart_data['RSI'])
        st.info(
            """
            **Reasoning:** RSI measures the speed and change of price movements to identify overbought or oversold conditions.
            - **Action:** A reading above 70 suggests the asset may be overbought and due for a correction. A reading below 30 suggests it may be oversold and poised for a rebound.
            """
        )
        st.subheader("Stochastic Oscillator")
        st.line_chart(chart_data[['Stoch_k', 'Stoch_d']])
        st.info(
            """
            **Reasoning:** This momentum indicator compares a specific closing price to a range of its prices over time.
            - **Action:** Like RSI, readings above 80 indicate overbought conditions, while readings below 20 indicate oversold conditions. Crossovers between the %K and %D lines can also be used as buy or sell signals.
            """
        )
    with tech_col2:
        st.subheader("MACD (Moving Average Convergence Divergence)")
        st.line_chart(chart_data[['MACD', 'MACD_Signal']])
        st.info(
            """
            **Reasoning:** MACD is a trend-following momentum indicator that shows the relationship between two moving averages.
            - **Action:** When the MACD line (blue) crosses above the Signal line (orange), it's a bullish signal, suggesting it may be a good time to buy. When it crosses below, it's a bearish signal.
            """
        )
        st.subheader("OBV (On-Balance Volume)")
        st.line_chart(chart_data['OBV'])
        st.info(
            """
            **Reasoning:** OBV uses volume flow to predict price changes. The idea is that volume precedes price.
            - **Action:** A rising OBV indicates positive volume pressure that can confirm an uptrend. A falling OBV suggests negative pressure that could signal a downtrend.
            """
        )
    st.subheader("Ichimoku Cloud")
    st.line_chart(chart_data[['Ichimoku_a', 'Ichimoku_b', 'Close']])
    st.info(
        """
        **Reasoning:** This is an all-in-one indicator that provides information on support, resistance, trend direction, and momentum.
        - **Action:** If the price is above the cloud, the overall trend is considered bullish. If the price is below the cloud, the trend is bearish. The cloud itself also acts as a dynamic zone of support or resistance.
        - **Ichimoku Cloud A vs. B:** The cloud is formed by the Senkou Span A (`Ichimoku_a`) and Senkou Span B (`Ichimoku_b`). When **A is above B**, the cloud is typically green and signals a **bullish trend**. When **B is above A**, the cloud is red and signals a **bearish trend**. The cloud's thickness indicates the strength of the trend.
        """
    )
else:
    st.warning(f"Could not load technical indicator data for {selected_coin}.")

# In dashboard_app.py, find and replace this entire section:

st.header(f"On-Chain & Fundamental Indicators for {selected_coin}")
if chart_data is not None:
    st.subheader("On-Chain & Market Indicators (from CoinGecko)")
    # --- THIS IS THE CORRECTED DESCRIPTION ---
    st.info(
        """
        **Reasoning:** These metrics provide a direct view of a blockchain's recent market activity. (Source: CoinGecko)
        - **Transaction Volume (24h):** The total value in USD of all transactions for this asset in the last 24 hours. High volume can help confirm the strength of a price trend.
        - **Circulating Supply:** The number of coins that are publicly available and circulating in the market. This is a key metric for calculating market capitalization and assessing scarcity.
        """
    )
    onchain_col1, onchain_col2 = st.columns(2)
    with onchain_col1:
        st.subheader("Transaction Volume (24h)")
        # Display as a single metric since it's a daily value, not historical
        if 'Transaction_Volume_24h' in chart_data.columns:
            latest_volume = chart_data['Transaction_Volume_24h'].iloc[-1]
            st.metric("Volume (USD)", f"${latest_volume:,.2f}")
        else:
            st.metric("Volume (USD)", "N/A")

    with onchain_col2:
        st.subheader("Circulating Supply")
        if 'Circulating_Supply' in chart_data.columns:
            latest_supply = chart_data['Circulating_Supply'].iloc[-1]
            st.metric("Supply", f"{latest_supply:,.0f} {selected_coin.split('-')[0]}")
        else:
            st.metric("Supply", "N/A")


    st.subheader("Fundamental Indicators (from CoinGecko)")
    st.info(
        """
        **Reasoning:** These scores assess the long-term viability, community health, and development activity of a project. (Source: CoinGecko)
        - **Market Cap Rank:** The project's rank relative to all other cryptocurrencies by market capitalization.
        - **Community Score:** A score based on Twitter followers, Telegram members, Reddit subscribers, etc.
        - **Developer Score:** A score based on GitHub activity like commits, stars, and forks.
        - **Sentiment:** The percentage of users who voted "Good" on CoinGecko.
        """
    )
    latest_fundamentals = chart_data.iloc[-1]
    fund_col1, fund_col2, fund_col3, fund_col4 = st.columns(4)
    fund_col1.metric("Market Cap Rank", f"#{latest_fundamentals.get('Market_Cap_Rank', 0):.0f}")
    fund_col2.metric("Community Score", f"{latest_fundamentals.get('Community_Score', 0):.1f}")
    fund_col3.metric("Developer Score", f"{latest_fundamentals.get('Developer_Score', 0):.1f}")
    fund_col4.metric("Sentiment (Up %)", f"{latest_fundamentals.get('Sentiment_Up_Percentage', 0):.1f}%")

else:
    st.warning(f"Could not load on-chain or fundamental data for {selected_coin}.")

# --- Raw Data Section ---
st.header("Raw Data Viewer")
st.subheader("Full Historical Forecast Data")
st.dataframe(format_numeric_columns(historical_df))
st.subheader(f"Full Daily Indicator Data for {selected_coin}")
if chart_data is not None:
    st.dataframe(format_numeric_columns(chart_data))

st.sidebar.markdown("---")
st.sidebar.info("This is for educational purposes only and is not financial advice.")
