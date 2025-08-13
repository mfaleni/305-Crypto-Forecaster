from dotenv import load_dotenv
load_dotenv() # Load variables from .env file FIRST

import streamlit as st
import pandas as pd
import os
import json
import numpy as np
from db_utils import load_forecast_results, update_feedback
# START: ADDED NEW IMPORT
from chart_analyst import analyze_bollinger_bands, analyze_rsi
# END: ADDED NEW IMPORT

# --- Page Configuration ---
st.set_page_config(page_title="305 Crypto Forecast", page_icon="📈", layout="wide")

# --- Robust Local Path Configuration ---
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

# START: ADDED NEW FIBONACCI CALCULATION FUNCTION
def calculate_fibonacci_levels(df: pd.DataFrame):
    """Calculates Fibonacci retracement levels for the given data."""
    if df.empty:
        return {}
    
    highest_high = df['High'].max()
    lowest_low = df['Low'].min()
    price_range = highest_high - lowest_low

    levels = {
        "Level 0% (High)": highest_high,
        "Level 23.6%": highest_high - (price_range * 0.236),
        "Level 38.2%": highest_high - (price_range * 0.382),
        "Level 50%": highest_high - (price_range * 0.5),
        "Level 61.8%": highest_high - (price_range * 0.618),
        "Level 100% (Low)": lowest_low,
    }
    return levels
# END: ADDED NEW FIBONACCI CALCULATION FUNCTION

# --- Main Application ---
st.title("📈 305 Crypto Forecast Dashboard")
st.markdown("An automated forecasting and sentiment analysis system for major cryptocurrencies.")

historical_df = get_historical_data()

if historical_df.empty:
    st.error("🚨 No forecast data found in the database. The daily analysis may not have run yet.")
    st.stop()

# --- Data Preparation ---
latest_date = historical_df['Date'].max()
latest_forecast_df = historical_df[historical_df['Date'] == latest_date].copy()

# --- Utility Function ---
def format_numeric_columns(df):
    formatted_df = df.copy()
    numeric_cols = [
        'Actual_Price', 'Prophet_Forecast', 'LSTM_Forecast', 'All_Time_High', 'RSI', 'MACD',
        'High', 'Low', 'Close', 'Open', 'Volume', 'SMA', 'EMA',
        'BB_High', 'BB_Low', 'Stoch_k', 'Stoch_d', 'OBV',
        'Ichimoku_a', 'Ichimoku_b', 'Transaction_Volume_24h', 'Circulating_Supply',
        'Market_Cap_Rank', 'Community_Score', 'Developer_Score', 'Sentiment_Up_Percentage',
        'Forecasted High', 'Funding_Rate', 'Open_Interest', 'Long_Short_Ratio',
        'MVRV_Ratio', 'Social_Dominance', 'Daily_Active_Addresses',
        'Galaxy_Score', 'Alt_Rank',
        # START: ADDED NEW COLS FOR FORMATTING
        'Leverage_Ratio', 'Futures_Volume_24h', 'Exchange_Supply_Ratio'
        # END: ADDED NEW COLS FOR FORMATTING
    ]
    for col in numeric_cols:
        if col in formatted_df.columns:
            try:
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
col1, col2, col3 = st.columns(3)
actual_price = pd.to_numeric(coin_forecast['Actual_Price'], errors='coerce')
all_time_high = pd.to_numeric(coin_forecast['All_Time_High'], errors='coerce')
sentiment_score = pd.to_numeric(coin_forecast['Sentiment_Score'], errors='coerce')
col1.metric("Actual Price", f"${actual_price:,.2f}" if pd.notna(actual_price) else "N/A")
col2.metric("All-Time High", f"${all_time_high:,.2f}" if pd.notna(all_time_high) else "N/A")
col3.metric("Sentiment Score", f"{sentiment_score:.2f}" if pd.notna(sentiment_score) else "N/A")

# --- START: UPGRADED AI ANALYSIS SECTION ---
st.header("Daily AI Analyst Report")
with st.container(border=True):
    title = coin_forecast.get('report_title', 'Analysis not available.')
    recap = coin_forecast.get('report_recap', '')
    bullish_case = coin_forecast.get('report_bullish', 'Bullish case not available.')
    bearish_case = coin_forecast.get('report_bearish', 'Bearish case not available.')
    hypothesis = coin_forecast.get('report_hypothesis', 'Hypothesis not available.')
    news_links_json = coin_forecast.get('analysis_news_links', '[]')
    
    st.subheader(title)
    st.caption(recap)
    
    st.markdown("---")

    col_bull, col_bear = st.columns(2)
    with col_bull:
        st.markdown("#### Bullish Case 🐂")
        st.markdown(bullish_case)
    with col_bear:
        st.markdown("#### Bearish Case 🐻")
        st.markdown(bearish_case)
    
    st.markdown("---")

    st.subheader("Analyst's Final Hypothesis")
    with st.container(border=True):
        st.markdown(hypothesis)

    st.subheader("Influential News")
    try:
        news_links = json.loads(news_links_json) if pd.notna(news_links_json) else []
        if news_links:
            for item in news_links:
                st.markdown(f"- [{item['title']}]({item['url']})")
        else:
            st.write("No specific news articles were identified as highly influential today.")
    except (json.JSONDecodeError, TypeError):
        st.write("Could not parse news links.")

    st.markdown("---")
    st.subheader("Provide Feedback on this Analysis")
    record_id = coin_forecast['id']
    current_feedback = coin_forecast.get('user_feedback')
    if pd.notna(current_feedback):
        st.success(f"Feedback previously saved: **{current_feedback}**")
    
    col_confirm, col_deny = st.columns(2)
    with col_confirm:
        if st.button("Confirm Analysis ✅", key=f"confirm_{record_id}"):
            if update_feedback(record_id, "Confirmed"):
                st.toast("Feedback 'Confirmed' saved!", icon="🎉")
                st.rerun()
            else:
                st.error("Failed to save feedback.")
    with col_deny:
        if st.button("Deny Analysis ❌", key=f"deny_{record_id}"):
            st.session_state[f'deny_clicked_{record_id}'] = True
    
    if st.session_state.get(f'deny_clicked_{record_id}'):
        with st.form(key=f"correction_form_{record_id}"):
            correction_text = st.text_area("What was wrong with the analysis? Please provide your correction.")
            submitted = st.form_submit_button("Submit Correction")
            
            if submitted:
                if update_feedback(record_id, "Denied", correction_text):
                    st.toast("Correction saved! Thank you.", icon="🙌")
                    st.session_state[f'deny_clicked_{record_id}'] = False
                    st.rerun()
                else:
                    st.error("Failed to save correction.")
# --- END: UPGRADED AI ANALYSIS SECTION ---


st.header("Professional Grade Market Indicators")
with st.container(border=True):
    st.subheader("Futures & Derivatives Data")
    # START: MODIFIED TO INCLUDE NEW METRICS
    cg_col1, cg_col2, cg_col3, cg_col4 = st.columns(4)
    funding_rate = pd.to_numeric(coin_forecast.get('Funding_Rate'), errors='coerce')
    open_interest = pd.to_numeric(coin_forecast.get('Open_Interest'), errors='coerce')
    long_short_ratio = pd.to_numeric(coin_forecast.get('Long_Short_Ratio'), errors='coerce')
    futures_volume = pd.to_numeric(coin_forecast.get('Futures_Volume_24h'), errors='coerce')
    
    cg_col1.metric("Funding Rate", f"{funding_rate:.4f}%" if pd.notna(funding_rate) else "N/A")
    cg_col2.metric("Open Interest", f"${open_interest:,.0f}" if pd.notna(open_interest) else "N/A")
    cg_col3.metric("Long/Short Ratio", f"{long_short_ratio:.2f}" if pd.notna(long_short_ratio) else "N/A")
    cg_col4.metric("Futures Volume (24h)", f"${futures_volume:,.0f}" if pd.notna(futures_volume) else "N/A")
    # END: MODIFIED TO INCLUDE NEW METRICS

    st.subheader("On-Chain & Social Metrics")
    # START: MODIFIED TO INCLUDE NEW METRICS
    san_col1, san_col2, san_col3, san_col4 = st.columns(4)
    mvrv = pd.to_numeric(coin_forecast.get('MVRV_Ratio'), errors='coerce')
    social_dom = pd.to_numeric(coin_forecast.get('Social_Dominance'), errors='coerce')
    daa = pd.to_numeric(coin_forecast.get('Daily_Active_Addresses'), errors='coerce')
    esr = pd.to_numeric(coin_forecast.get('Exchange_Supply_Ratio'), errors='coerce')

    san_col1.metric("MVRV Ratio", f"{mvrv:.2f}" if pd.notna(mvrv) else "N/A")
    san_col2.metric("Social Dominance", f"{social_dom:.2f}%" if pd.notna(social_dom) else "N/A")
    san_col3.metric("Daily Active Addresses", f"{daa:,.0f}" if pd.notna(daa) else "N/A")
    san_col4.metric("Exchange Supply Ratio", f"{esr:.2f}" if pd.notna(esr) else "N/A (Placeholder)")
    # END: MODIFIED TO INCLUDE NEW METRICS

    st.subheader("Social Intelligence (from LunarCrush)")
    lc_col1, lc_col2 = st.columns(2)
    galaxy_score = pd.to_numeric(coin_forecast.get('Galaxy_Score'), errors='coerce')
    alt_rank = pd.to_numeric(coin_forecast.get('Alt_Rank'), errors='coerce')
    lc_col1.metric("Galaxy Score™", f"{galaxy_score:.1f}/100" if pd.notna(galaxy_score) else "N/A")
    lc_col2.metric("AltRank™", f"#{alt_rank:.0f}" if pd.notna(alt_rank) else "N/A")

st.header("5-Day High Forecast vs. Historical Highs")
# This section remains unchanged
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
    except (json.JSONDecodeError, TypeError):
        st.error("Could not parse the 5-day forecast data.")
else:
    st.warning(f"Could not load 5-day forecast data for {selected_coin}.")

st.header(f"Technical Indicators for {selected_coin}")
if chart_data is not None:
    # START: UPGRADED BOLLINGER BANDS SECTION
    st.subheader("Price, Moving Averages, & Bollinger Bands")
    st.line_chart(chart_data[['Close', 'SMA', 'EMA', 'BB_High', 'BB_Low']])
    st.info(f"**Analysis:** {analyze_bollinger_bands(chart_data)}") # DYNAMIC ANALYSIS
    # END: UPGRADED BOLLINGER BANDS SECTION
    
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        # START: UPGRADED RSI SECTION
        st.subheader("RSI (Relative Strength Index)")
        st.line_chart(chart_data['RSI'])
        st.info(f"**Analysis:** {analyze_rsi(chart_data)}") # DYNAMIC ANALYSIS
        # END: UPGRADED RSI SECTION
        
        st.subheader("Stochastic Oscillator")
        st.line_chart(chart_data[['Stoch_k', 'Stoch_d']])
        # Placeholder for dynamic Stochastic analysis
        st.info("**Analysis:** Readings above 80 indicate overbought conditions, while below 20 indicate oversold.")

    with tech_col2:
        st.subheader("MACD (Moving Average Convergence Divergence)")
        st.line_chart(chart_data[['MACD', 'MACD_Signal']])
        # Placeholder for dynamic MACD analysis
        st.info("**Analysis:** When the MACD line crosses above the Signal line, it's a bullish signal.")

        st.subheader("OBV (On-Balance Volume)")
        st.line_chart(chart_data['OBV'])
        # Placeholder for dynamic OBV analysis
        st.info("**Analysis:** A rising OBV indicates positive volume pressure that can confirm an uptrend.")

    # START: UPGRADED PRICE CHART WITH FIBONACCI LEVELS
    st.subheader("Price Chart with Fibonacci Levels")
    fib_levels = calculate_fibonacci_levels(chart_data)
    
    # Create a new DataFrame for plotting that includes the price and Fib levels
    plot_df = chart_data[['Close']].copy()
    for name, level in fib_levels.items():
        plot_df[name] = level
        
    st.line_chart(plot_df)
    st.info(f"""
        **Analysis:** The Fibonacci levels are key potential areas of support and resistance. 
        The market will often see price react around these levels. 
        Currently, the key support is at the **{list(fib_levels.keys())[4]}** (${fib_levels[list(fib_levels.keys())[4]]:,.2f}) and 
        the key resistance is at the **{list(fib_levels.keys())[1]}** (${fib_levels[list(fib_levels.keys())[1]]:,.2f}).
    """)
    # END: UPGRADED PRICE CHART WITH FIBONACCI LEVELS

else:
    st.warning(f"Could not load technical indicator data for {selected_coin}.")

# ... (The "On-Chain & Fundamental Indicators" and "Raw Data Viewer" sections remain unchanged) ...

st.header(f"On-Chain & Fundamental Indicators for {selected_coin}")
if chart_data is not None:
    st.subheader("On-Chain & Market Indicators (from CoinGecko)")
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
        - **Community Score:** A score based on social media activity.
        - **Developer Score:** A score based on GitHub activity.
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

st.header("Raw Data Viewer")
st.subheader("Full Historical Forecast Data")
st.dataframe(format_numeric_columns(historical_df))
st.subheader(f"Full Daily Indicator Data for {selected_coin}")
if chart_data is not None:
    st.dataframe(format_numeric_columns(chart_data))

st.sidebar.markdown("---")
st.sidebar.info("This is for educational purposes only and is not financial advice.")