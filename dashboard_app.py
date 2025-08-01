import streamlit as st
import pandas as pd
import os
import time

# Import our project modules
from data_utils import fetch_data
from daily_runner import run_daily_analysis, RESULTS_FILE

# --- Page Configuration ---
st.set_page_config(
    page_title="305 Crypto Forecast",
    page_icon="📈",
    layout="wide"
)

# --- Data Loading and Analysis ---
# Check if the results file exists and if it's recent (less than 24 hours old)
should_run_analysis = False
if not os.path.exists(RESULTS_FILE):
    st.warning("Data file not found. A fresh analysis will be run.")
    should_run_analysis = True
else:
    file_age = time.time() - os.path.getmtime(RESULTS_FILE)
    if file_age > 86400: # 86400 seconds = 24 hours
        st.info("Data is older than 24 hours. Running a fresh analysis.")
        should_run_analysis = True

# If needed, run the full analysis and show progress
if should_run_analysis:
    with st.spinner("🚀 Running fresh analysis... This may take a few minutes."):
        run_daily_analysis()
    st.success("✅ Analysis complete! Displaying the latest data.")
    # Rerun the script to load the new data into the app
    st.experimental_rerun()


# --- Main Application ---
st.title("📈 305 Crypto Forecast Dashboard")
st.markdown("An automated forecasting and sentiment analysis system for major cryptocurrencies.")

# --- Load Forecast Data ---
try:
    forecast_df = pd.read_csv(RESULTS_FILE)
except FileNotFoundError:
    st.error("Analysis failed to generate data. Please check the logs.")
    st.stop()

# --- Caching Function for charts ---
@st.cache_data
def load_chart_data(ticker):
    return fetch_data(ticker)

# --- Sidebar & Main Content ---
st.sidebar.header("Dashboard Options")
selected_coin = st.sidebar.selectbox(
    "Select a Cryptocurrency",
    forecast_df['Coin'].unique()
)

# --- Load Comprehensive Data for Charts ---
chart_data = load_chart_data(selected_coin)

# --- Main Tabbed Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Technical Analysis", "On-Chain & Fundamentals", "Raw Data"])

with tab1:
    st.header(f"Today's Forecast for {selected_coin}")

    # Filter the dataframe for the selected coin
    coin_forecast = forecast_df[forecast_df['Coin'] == selected_coin].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Actual Price", f"${coin_forecast['Actual_Price']:,.2f}")
    col2.metric("Prophet Forecast", f"${coin_forecast['Prophet_Forecast']:,.2f}")
    col3.metric("LSTM Forecast", f"${coin_forecast['LSTM_Forecast']:,.2f}")
    col4.metric("Sentiment Score", f"{coin_forecast['Sentiment_Score']:.2f}")
    
    if not chart_data.empty:
        st.subheader("Price History (Last 180 Days)")
        st.line_chart(chart_data['Close'])
    else:
        st.warning(f"Could not load chart data for {selected_coin}.")

with tab2:
    st.header(f"Technical Indicators for {selected_coin}")
    if not chart_data.empty:
        st.subheader("Price, Moving Averages, & Bollinger Bands")
        st.line_chart(chart_data[['Close', 'SMA', 'EMA', 'BB_High', 'BB_Low']])
        st.info(
            """
            - **SMA (Simple Moving Average):** The average price over a specified period.
            - **EMA (Exponential Moving Average):** Gives more weight to recent prices.
            - **Bollinger Bands:** A measure of volatility.
            """
        )
        
        tech_col1, tech_col2 = st.columns(2)
        with tech_col1:
            st.subheader("RSI")
            st.line_chart(chart_data['RSI'])
            st.markdown("*Values **>70** may indicate overbought, **<30** oversold.*")
            st.subheader("Stochastic Oscillator")
            st.line_chart(chart_data[['Stoch_k', 'Stoch_d']])
            st.markdown("*Identifies overbought/oversold conditions.*")
        with tech_col2:
            st.subheader("MACD")
            st.line_chart(chart_data[['MACD', 'MACD_Signal']])
            st.markdown("*A trend-following momentum indicator.*")
            st.subheader("On-Balance Volume (OBV)")
            st.line_chart(chart_data['OBV'])
            st.markdown("*Uses volume flow to predict price changes.*")
        st.subheader("Ichimoku Cloud")
        st.line_chart(chart_data[['Ichimoku_a', 'Ichimoku_b', 'Close']])
        st.markdown("""
        *Shows support/resistance and momentum. When price is above the cloud, trend is bullish.*
        - **Ichimoku A:** The faster leading span.
        - **Ichimoku B:** The slower leading span.
        """)
    else:
        st.warning(f"Could not load technical indicator data for {selected_coin}.")

with tab3:
    st.header(f"On-Chain & Fundamental Indicators for {selected_coin}")
    if not chart_data.empty:
        st.subheader("On-Chain Indicators (Simulated)")
        st.info("Insights into blockchain network health and activity (values are simulated).")
        onchain_col1, onchain_col2 = st.columns(2)
        with onchain_col1:
            st.subheader("Active Addresses")
            st.bar_chart(chart_data['Active_Addresses'])
            st.subheader("Total Value Locked (TVL)")
            st.area_chart(chart_data['TVL'])
        with onchain_col2:
            st.subheader("Transaction Volume")
            st.bar_chart(chart_data['Transaction_Volume'])
            st.subheader("Realized PnL")
            st.area_chart(chart_data['Realized_PnL'])

        st.subheader("Fundamental Indicators (Simulated)")
        st.info("Scores (1-10) assessing the intrinsic value of the project (values are simulated).")
        latest_fundamentals = chart_data.iloc[-1]
        fund_col1, fund_col2, fund_col3, fund_col4, fund_col5 = st.columns(5)
        fund_col1.metric("Token Utility", f"{latest_fundamentals['Token_Utility']:.1f}")
        fund_col2.metric("Adoption Rate", f"{latest_fundamentals['Adoption_Rate']:.1f}")
        fund_col3.metric("Team Score", f"{latest_fundamentals['Team_Score']:.1f}")
        fund_col4.metric("Tokenomics", f"{latest_fundamentals['Tokenomics_Score']:.1f}")
        fund_col5.metric("Regulatory Risk", f"{latest_fundamentals['Regulatory_Risk']:.1f}", delta_color="inverse")
    else:
        st.warning(f"Could not load on-chain or fundamental data for {selected_coin}.")

with tab4:
    st.header("Raw Data Viewer")
    st.subheader("Forecast Summary Data")
    st.dataframe(forecast_df)
    st.subheader(f"Full Indicator Data for {selected_coin}")
    if not chart_data.empty:
        st.dataframe(chart_data)
    else:
        st.warning(f"No indicator data available for {selected_coin}.")

st.sidebar.markdown("---")
st.sidebar.info("This dashboard is for educational purposes only and is not financial advice.")
