from dotenv import load_dotenv
load_dotenv() # Load variables from .env file FIRST

import streamlit as st
import pandas as pd
import os
import json
import numpy as np
from db_utils import load_forecast_results, update_feedback
# RETAINED: Existing imports for dynamic chart analysis
from chart_analyst import analyze_bollinger_bands, analyze_rsi

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

# RETAINED: Existing Fibonacci Calculation Function
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

# --- Utility Function (UPDATED) ---
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
        # RETAINED: Existing advanced metrics
        'Leverage_Ratio', 'Futures_Volume_24h', 'Exchange_Supply_Ratio',
        # NEW: Added Trade Recommendation columns
        'trade_tp1', 'trade_tp2', 'trade_sl', 'trade_confidence'
    ]
    for col in numeric_cols:
        if col in formatted_df.columns:
            try:
                 # Specific formatting for confidence score (percentage)
                if col == 'trade_confidence':
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) and isinstance(x, (float)) else x)
                else:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x)
            except (ValueError, TypeError):
                pass
    return formatted_df

# --- Main Application ---
st.title("📈 305 Crypto Forecast Dashboard")
st.markdown("An automated forecasting, strategy, and sentiment analysis system.")

historical_df = get_historical_data()

if historical_df.empty:
    st.error("🚨 No forecast data found in the database. The daily analysis may not have run yet.")
    st.stop()

# --- Data Preparation ---
# Ensure Date column is datetime if loaded as object
if 'Date' in historical_df.columns and historical_df['Date'].dtype == 'object':
    historical_df['Date'] = pd.to_datetime(historical_df['Date'])

latest_date = historical_df['Date'].max()
latest_forecast_df = historical_df[historical_df['Date'] == latest_date].copy()

# --- Sidebar & Main Content ---
st.sidebar.header("Dashboard Options")

if not latest_forecast_df.empty:
    selected_coin = st.sidebar.selectbox("Select a Cryptocurrency", latest_forecast_df['Coin'].unique())
    chart_data = load_chart_data(selected_coin)
    # Ensure the specific coin forecast exists
    coin_forecast_series = latest_forecast_df[latest_forecast_df['Coin'] == selected_coin]
    if not coin_forecast_series.empty:
        coin_forecast = coin_forecast_series.iloc[0]
    else:
        st.warning(f"No data found for {selected_coin} on the latest date.")
        st.stop()
else:
    st.warning("No data available for the latest date.")
    st.stop()


# --- Main Page Layout ---
st.header(f"Today's Overview for {selected_coin}")
col1, col2, col3 = st.columns(3)
actual_price = pd.to_numeric(coin_forecast['Actual_Price'], errors='coerce')
all_time_high = pd.to_numeric(coin_forecast['All_Time_High'], errors='coerce')
sentiment_score = pd.to_numeric(coin_forecast['Sentiment_Score'], errors='coerce')
col1.metric("Actual Price", f"${actual_price:,.2f}" if pd.notna(actual_price) else "N/A")
col2.metric("All-Time High", f"${all_time_high:,.2f}" if pd.notna(all_time_high) else "N/A")
col3.metric("Sentiment Score", f"{sentiment_score:.2f}" if pd.notna(sentiment_score) else "N/A")

# +++ START: NEW AI TRADE RECOMMENDATION SECTION +++
st.header("⚡ AI Strategy Agent: Trade Setup (24-72h Horizon)")

trade_action = coin_forecast.get('trade_action', 'HOLD')
trade_rationale = coin_forecast.get('trade_rationale', 'No rationale provided.')
trade_confidence = pd.to_numeric(coin_forecast.get('trade_confidence'), errors='coerce')
trade_entry = coin_forecast.get('trade_entry_range', 'N/A')
trade_tp1 = pd.to_numeric(coin_forecast.get('trade_tp1'), errors='coerce')
trade_tp2 = pd.to_numeric(coin_forecast.get('trade_tp2'), errors='coerce')
trade_sl = pd.to_numeric(coin_forecast.get('trade_sl'), errors='coerce')

# Format confidence display
confidence_display = f"{trade_confidence*100:.1f}%" if pd.notna(trade_confidence) else "N/A"

# Determine the visual style and banner text based on the action
if trade_action == "BUY":
    status_type = "success"
    icon = "🐂"
    action_text = f"BUY SIGNAL (Confidence: {confidence_display})"
elif trade_action == "SELL":
    status_type = "error"
    icon = "🐻"
    action_text = f"SELL SIGNAL (Confidence: {confidence_display})"
else:
    status_type = "info"
    icon = "⏸️"
    action_text = f"HOLD (No High-Probability Setup Identified)"

# Display the recommendation
with st.container(border=True):
    # Display the main action banner using Streamlit status elements
    if status_type == "success":
        st.success(f"**{icon} {action_text}**")
    elif status_type == "error":
        st.error(f"**{icon} {action_text}**")
    else:
        st.info(f"**{icon} {action_text}**")

    # Display the rationale
    st.markdown(f"**Strategy Rationale:** {trade_rationale}")
    
    # Display the trade parameters only if the action is BUY or SELL
    if trade_action in ["BUY", "SELL"]:
        st.markdown("---")
        st.subheader("Trade Parameters")
        
        col_entry, col_tp1, col_sl, col_tp2 = st.columns(4)
        
        col_entry.metric("Entry Range", trade_entry)

        # Take Profit 1 (Reward Calculation)
        if pd.notna(trade_tp1) and pd.notna(actual_price) and actual_price != 0:
            reward_percent = ((trade_tp1 - actual_price) / actual_price) * 100 if trade_action == "BUY" else ((actual_price - trade_tp1) / actual_price) * 100
            col_tp1.metric("Target Profit 1 (TP1)", f"${trade_tp1:,.2f}", delta=f"{reward_percent:.2f}% Gain")
        else:
            col_tp1.metric("Target Profit 1 (TP1)", f"${trade_tp1:,.2f}" if pd.notna(trade_tp1) else "N/A")
        
        # Stop Loss (Risk Calculation)
        if pd.notna(trade_sl) and pd.notna(actual_price) and actual_price != 0:
            # Risk is the absolute percentage loss if SL is hit
            risk_percent = abs(((trade_sl - actual_price) / actual_price) * 100)
            col_sl.metric("Stop Loss (SL)", f"${trade_sl:,.2f}", delta=f"-{risk_percent:.2f}% Risk", delta_color="inverse")
        else:
            col_sl.metric("Stop Loss (SL)", f"${trade_sl:,.2f}" if pd.notna(trade_sl) else "N/A")

        # Take Profit 2
        col_tp2.metric("Target Profit 2 (TP2)", f"${trade_tp2:,.2f}" if pd.notna(trade_tp2) and trade_tp2 > 0 else "N/A")

# +++ END: NEW AI TRADE RECOMMENDATION SECTION +++


# --- RETAINED: UPGRADED AI ANALYSIS SECTION ---
st.header("Daily AI Analyst Report")
with st.container(border=True):
    title = coin_forecast.get('report_title', 'Analysis not available.')
    recap = coin_forecast.get('report_recap', '')
    bullish_case = coin_forecast.get('report_bullish', 'Bullish case not available.')
    bearish_case = coin_forecast.get('report_bearish', 'Bearish case not available.')
    hypothesis = coin_forecast.get('report_hypothesis', 'Hypothesis not available.')
    # Note: Assuming 'analysis_news_links' is still the source for news in this report
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
                # Check if 'title' and 'url' keys exist
                if isinstance(item, dict) and 'title' in item and 'url' in item:
                    st.markdown(f"- [{item['title']}]({item['url']})")
        else:
            st.write("No specific news articles were identified as highly influential today.")
    except (json.JSONDecodeError, TypeError):
        st.write("Could not parse news links.")

    st.markdown("---")
    st.subheader("Provide Feedback on this Analysis")
    # Ensure record_id is handled correctly
    record_id = coin_forecast.get('id')
    if pd.notna(record_id):
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
    else:
        st.error("Cannot provide feedback. Record ID missing.")
# --- END: UPGRADED AI ANALYSIS SECTION ---


st.header("Professional Grade Market Indicators")
with st.container(border=True):
    st.subheader("Futures & Derivatives Data")
    # RETAINED: Existing advanced metrics layout
    cg_col1, cg_col2, cg_col3, cg_col4 = st.columns(4)
    funding_rate = pd.to_numeric(coin_forecast.get('Funding_Rate'), errors='coerce')
    open_interest = pd.to_numeric(coin_forecast.get('Open_Interest'), errors='coerce')
    long_short_ratio = pd.to_numeric(coin_forecast.get('Long_Short_Ratio'), errors='coerce')
    futures_volume = pd.to_numeric(coin_forecast.get('Futures_Volume_24h'), errors='coerce')
    
    cg_col1.metric("Funding Rate", f"{funding_rate:.4f}%" if pd.notna(funding_rate) else "N/A")
    cg_col2.metric("Open Interest", f"${open_interest:,.0f}" if pd.notna(open_interest) else "N/A")
    cg_col3.metric("Long/Short Ratio", f"{long_short_ratio:.2f}" if pd.notna(long_short_ratio) else "N/A")
    cg_col4.metric("Futures Volume (24h)", f"${futures_volume:,.0f}" if pd.notna(futures_volume) else "N/A")

    st.subheader("On-Chain & Social Metrics")
    # RETAINED: Existing advanced metrics layout
    san_col1, san_col2, san_col3, san_col4 = st.columns(4)
    mvrv = pd.to_numeric(coin_forecast.get('MVRV_Ratio'), errors='coerce')
    social_dom = pd.to_numeric(coin_forecast.get('Social_Dominance'), errors='coerce')
    daa = pd.to_numeric(coin_forecast.get('Daily_Active_Addresses'), errors='coerce')
    esr = pd.to_numeric(coin_forecast.get('Exchange_Supply_Ratio'), errors='coerce')

    san_col1.metric("MVRV Ratio", f"{mvrv:.2f}" if pd.notna(mvrv) else "N/A")
    san_col2.metric("Social Dominance", f"{social_dom:.2f}%" if pd.notna(social_dom) else "N/A")
    san_col3.metric("Daily Active Addresses", f"{daa:,.0f}" if pd.notna(daa) else "N/A")
    # Retaining the placeholder note if it's still relevant
    san_col4.metric("Exchange Supply Ratio", f"{esr:.2f}" if pd.notna(esr) else "N/A (Placeholder)")

    st.subheader("Social Intelligence (from LunarCrush)")
    lc_col1, lc_col2 = st.columns(2)
    galaxy_score = pd.to_numeric(coin_forecast.get('Galaxy_Score'), errors='coerce')
    alt_rank = pd.to_numeric(coin_forecast.get('Alt_Rank'), errors='coerce')
    lc_col1.metric("Galaxy Score™", f"{galaxy_score:.1f}/100" if pd.notna(galaxy_score) else "N/A")
    lc_col2.metric("AltRank™", f"#{int(alt_rank)}" if pd.notna(alt_rank) else "N/A")

st.header("5-Day High Forecast vs. Historical Highs")
# This section remains unchanged
if chart_data is not None and 'High_Forecast_5_Day' in coin_forecast and pd.notna(coin_forecast['High_Forecast_5_Day']):
    historical_highs = chart_data[['High']].tail(5)
    # Ensure index is datetime for consistent formatting
    historical_highs.index = pd.to_datetime(historical_highs.index).strftime('%Y-%m-%d')
    try:
        forecast_data = json.loads(str(coin_forecast['High_Forecast_5_Day']))
        if forecast_data:
            forecast_df_highs = pd.DataFrame(forecast_data)
            # Check if 'ds' column exists and process
            if 'ds' in forecast_df_highs.columns:
                forecast_df_highs['ds'] = pd.to_datetime(forecast_df_highs['ds']).dt.strftime('%Y-%m-%d')
                forecast_df_highs = forecast_df_highs.rename(columns={'ds': 'Date', 'yhat': 'Forecasted High'}).set_index('Date')
                # Rename historical column for clarity
                combined_df = pd.concat([historical_highs.rename(columns={'High': 'Historical High'}), forecast_df_highs['Forecasted High']], axis=1)
                st.bar_chart(combined_df)
                st.dataframe(format_numeric_columns(combined_df.reset_index()))
            else:
                st.warning("Forecast data structure is unexpected (missing 'ds' column).")
        else:
            st.warning("Forecast data is available but empty.")
    except (json.JSONDecodeError, TypeError) as e:
        st.error(f"Could not parse the 5-day forecast data: {e}")
else:
    st.warning(f"Could not load 5-day forecast data or chart data for {selected_coin}.")

st.header(f"Technical Indicators for {selected_coin}")
if chart_data is not None:
    # RETAINED: UPGRADED BOLLINGER BANDS SECTION
    st.subheader("Price, Moving Averages, & Bollinger Bands")
    st.line_chart(chart_data[['Close', 'SMA', 'EMA', 'BB_High', 'BB_Low']])
    st.info(f"**Analysis:** {analyze_bollinger_bands(chart_data)}") # DYNAMIC ANALYSIS
    
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        # RETAINED: UPGRADED RSI SECTION
        st.subheader("RSI (Relative Strength Index)")
        st.line_chart(chart_data['RSI'])
        st.info(f"**Analysis:** {analyze_rsi(chart_data)}") # DYNAMIC ANALYSIS
        
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

    # RETAINED: UPGRADED PRICE CHART WITH FIBONACCI LEVELS
    st.subheader("Price Chart with Fibonacci Levels")
    fib_levels = calculate_fibonacci_levels(chart_data)
    
    if fib_levels:
        # Create a new DataFrame for plotting that includes the price and Fib levels
        plot_df = chart_data[['Close']].copy()
        for name, level in fib_levels.items():
            plot_df[name] = level
            
        st.line_chart(plot_df)
        
        # Ensure the Fibonacci keys exist before accessing them for the info box
        fib_keys = list(fib_levels.keys())
        if len(fib_keys) >= 5:
            st.info(f"""
                **Analysis:** The Fibonacci levels are key potential areas of support and resistance. 
                The market will often see price react around these levels. 
                Currently, the key support is at the **{fib_keys[4]}** (${fib_levels[fib_keys[4]]:,.2f}) and 
                the key resistance is at the **{fib_keys[1]}** (${fib_levels[fib_keys[1]]:,.2f}).
            """)
        else:
             st.info("Fibonacci levels calculated. Used to identify potential support and resistance.")
    else:
        st.warning("Could not calculate Fibonacci levels (e.g., insufficient data range).")


else:
    st.warning(f"Could not load technical indicator data for {selected_coin}.")

# ... (The "On-Chain & Fundamental Indicators" and "Raw Data Viewer" sections remain unchanged) ...

st.header(f"On-Chain & Fundamental Indicators for {selected_coin}")
if chart_data is not None and not chart_data.empty:
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
            st.metric("Volume (USD)", f"${latest_volume:,.2f}" if pd.notna(latest_volume) else "N/A")
        else:
            st.metric("Volume (USD)", "N/A")
    with onchain_col2:
        st.subheader("Circulating Supply")
        if 'Circulating_Supply' in chart_data.columns:
            latest_supply = chart_data['Circulating_Supply'].iloc[-1]
            st.metric("Supply", f"{latest_supply:,.0f} {selected_coin.split('-')[0]}" if pd.notna(latest_supply) else "N/A")
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

    # Robust handling of fundamental data
    mc_rank = latest_fundamentals.get('Market_Cap_Rank', np.nan)
    comm_score = latest_fundamentals.get('Community_Score', np.nan)
    dev_score = latest_fundamentals.get('Developer_Score', np.nan)
    sent_up = latest_fundamentals.get('Sentiment_Up_Percentage', np.nan)

    fund_col1.metric("Market Cap Rank", f"#{int(mc_rank)}" if pd.notna(mc_rank) else "N/A")
    fund_col2.metric("Community Score", f"{comm_score:.1f}" if pd.notna(comm_score) else "N/A")
    fund_col3.metric("Developer Score", f"{dev_score:.1f}" if pd.notna(dev_score) else "N/A")
    fund_col4.metric("Sentiment (Up %)", f"{sent_up:.1f}%" if pd.notna(sent_up) else "N/A")
else:
    st.warning(f"Could not load on-chain or fundamental data for {selected_coin}.")

st.header("Raw Data Viewer")
st.subheader("Full Historical Forecast Data")
# Apply formatting to the historical data viewer as well
st.dataframe(format_numeric_columns(historical_df))

st.subheader(f"Full Daily Indicator Data for {selected_coin}")
if chart_data is not None:
    st.dataframe(format_numeric_columns(chart_data))

st.sidebar.markdown("---")
st.sidebar.info("This is for educational purposes only and is not financial advice.")