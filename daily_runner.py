import pandas as pd
from datetime import datetime
import numpy as np
import os
import json
import openai
from frozendict import frozendict
from dotenv import load_dotenv
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Environment and Keys ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

# --- Robust Key Check ---
# This checks all required API keys. We rely on db_utils.py for intelligent DB connection handling.
required_keys = [
    "OPENAI_API_KEY", "NEWS_API_KEY", "SANTIMENT_API_KEY", 
    "LUNARCRUSH_API_KEY", "COINGECKO_API_KEY"
]
missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    logger.warning(f" ⚠️  [WARN] The following API keys are missing: {', '.join(missing_keys)}. Some data sources may fail.")
    # We don't exit immediately, allowing the script to proceed if possible, but logging the issue.

# --- Module Imports ---
try:
    from data_utils import fetch_data
    from forecasting import prophet_forecast, lstm_forecast, prophet_forecast_highs
    from sentiment import get_news_sentiment
    from db_utils import init_db, save_forecast_results
    # analyst.py generates the existing reports
    from analyst import get_daily_analysis 
    # NEW: IMPORT THE STRATEGY AGENT
    from strategy_agent import get_trade_recommendation 
except ImportError as e:
    logger.error(f" ❌  [FATAL] Failed to import a required module: {e}. Exiting.")
    exit(1)

# --- Configuration ---
# Adjust COINS as needed for your specific setup.
COINS = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "XRP-USD": "XRP"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

def default_json_serializer(obj):
    """Helper for JSON serialization of complex types."""
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    if isinstance(obj, frozendict): return dict(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def run_daily_analysis():
    logger.info(" ✅  [START] Kicking off daily crypto forecasting run...")
    
    # Initialize DB connection and schema
    try:
        init_db()
    except Exception as e:
        logger.error(f" ❌  [FATAL] Database initialization failed: {e}. Exiting.")
        exit(1)

    # Use a consistent timestamp for the entire run
    run_time = datetime.now()
    os.makedirs(DATA_DIR, exist_ok=True)
    all_results = []

    for ticker, name in COINS.items():
        logger.info(f"\nProcessing {ticker} ({name})...")
        try:
            market_data = fetch_data(ticker)
            # Check for minimum data required (e.g., 61 days for LSTM lookback)
            if market_data.empty or len(market_data) < 61:
                logger.warning(f"   [WARN] Insufficient data for {ticker}. Skipping.")
                continue
            
            # Save detailed data for the dashboard
            detailed_data_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
            market_data.to_csv(detailed_data_path)
            
            latest_data = market_data.iloc[-1]
            
            # Run forecasts and sentiment analysis
            prophet_price = prophet_forecast(market_data.copy())
            lstm_price = lstm_forecast(market_data.copy())
            high_forecasts_list = prophet_forecast_highs(market_data.copy(), periods=5)
            sentiment_score, top_headlines = get_news_sentiment(coin_ticker=ticker, coin_name=name, api_key=news_api_key)

            # Prepare a comprehensive briefing for the AI (Used by Analyst and Strategist)
            # Ensure ALL relevant data points are included for the AI prompts.
            daily_briefing_data = {
                "coin_name": name,
                "actual_price": latest_data.get("Close", 0.0),
                "prophet_forecast": prophet_price,
                "lstm_forecast": lstm_price,
                "sentiment_score": sentiment_score,
                "rsi": latest_data.get("RSI", 0.0),
                "macd": latest_data.get("MACD", 0.0),
                "funding_rate": latest_data.get("Funding_Rate", 0.0),
                "open_interest": latest_data.get("Open_Interest", 0.0),
                "long_short_ratio": latest_data.get("Long_Short_Ratio", 0.0),
                "mvrv_ratio": latest_data.get("MVRV_Ratio", 0.0),
                "social_dominance": latest_data.get("Social_Dominance", 0.0),
                "daily_active_addresses": latest_data.get("Daily_Active_Addresses", 0.0),
                "galaxy_score": latest_data.get("Galaxy_Score", 0.0),
                "alt_rank": latest_data.get("Alt_Rank", 0.0),
                # Include advanced metrics (ensure these exist in your fetch_data output)
                "Leverage_Ratio": latest_data.get("Leverage_Ratio", 0.0),
                "Futures_Volume_24h": latest_data.get("Futures_Volume_24h", 0.0),
                "Exchange_Supply_Ratio": latest_data.get("Exchange_Supply_Ratio", 0.0),
                "top_headlines": top_headlines
            }

            # 1. Get the existing AI Analysis (Descriptive Report)
            analysis_results = get_daily_analysis(daily_briefing_data)

            # 2. Get the NEW AI Trade Recommendation (Prescriptive Strategy)
            trade_recommendation = get_trade_recommendation(daily_briefing_data)

            # Assemble the final, complete record for the database
            result = {
                "Date": run_time, # Use the consistent timestamp
                "Coin": ticker,
                
                # --- Standard Market Data & Forecasts ---
                "Actual_Price": latest_data.get("Close", 0.0),
                "Prophet_Forecast": prophet_price,
                "LSTM_Forecast": lstm_price,
                "Sentiment_Score": sentiment_score,
                "RSI": latest_data.get("RSI", 0.0),
                "MACD": latest_data.get("MACD", 0.0),
                "All_Time_High": latest_data.get("All_Time_High_Real", 0.0),
                "High_Forecast_5_Day": json.dumps(high_forecasts_list, default=default_json_serializer),
                
                # --- Professional Data Sources (CoinGlass, Santiment, LunarCrush) ---
                "Funding_Rate": latest_data.get("Funding_Rate", 0.0),
                "Open_Interest": latest_data.get("Open_Interest", 0.0),
                "Long_Short_Ratio": latest_data.get("Long_Short_Ratio", 0.0),
                "MVRV_Ratio": latest_data.get("MVRV_Ratio", 0.0),
                "Social_Dominance": latest_data.get("Social_Dominance", 0.0),
                "Daily_Active_Addresses": latest_data.get("Daily_Active_Addresses", 0.0),
                "Galaxy_Score": latest_data.get("Galaxy_Score", 0.0),
                "Alt_Rank": latest_data.get("Alt_Rank", 0.0),
                "Exchange_Net_Flow": latest_data.get("Exchange_Net_Flow", 0.0),

                # --- Advanced Metrics (Retained from your existing setup) ---
                "Leverage_Ratio": latest_data.get("Leverage_Ratio", 0.0),
                "Futures_Volume_24h": latest_data.get("Futures_Volume_24h", 0.0),
                "Exchange_Supply_Ratio": latest_data.get("Exchange_Supply_Ratio", 0.0),

                # --- AI Analysis (Basic summary from the Guide documentation) ---
                "analysis_summary": analysis_results.get("summary"),
                "analysis_hypothesis": analysis_results.get("hypothesis"),
                "analysis_news_links": analysis_results.get("news_links"),
                "user_feedback": None,
                "user_correction": None,

                # --- AI Report (Your existing custom fields - RETAINED) ---
                "report_title": analysis_results.get("report_title"),
                "report_recap": analysis_results.get("report_recap"),
                "report_bullish": analysis_results.get("report_bullish"),
                "report_bearish": analysis_results.get("report_bearish"),
                "report_hypothesis": analysis_results.get("report_hypothesis"),
                
                # --- NEW: AI Trade Recommendations (from Strategy Agent) ---
                "trade_action": trade_recommendation.get("action"),
                "trade_entry_range": trade_recommendation.get("entry_range"),
                "trade_tp1": trade_recommendation.get("tp1"),
                "trade_tp2": trade_recommendation.get("tp2"),
                "trade_sl": trade_recommendation.get("sl"),
                "trade_confidence": trade_recommendation.get("confidence"),
                "trade_rationale": trade_recommendation.get("rationale")
            }
            all_results.append(result)
        except Exception as e:
            logger.error(f" ❌  [ERROR] An unexpected error occurred while processing {ticker}: {e}", exc_info=True)
            continue
            
    logger.info("\n ✅  [FINISH] Daily processing complete.")
    if all_results:
        results_df = pd.DataFrame(all_results)
        try:
            save_forecast_results(results_df)
        except Exception as e:
            logger.error(f" ❌  [ERROR] Failed to save results to the database: {e}")
    else:
        logger.warning("\n[WARN] No results were generated. Database not updated.")

if __name__ == "__main__":
    run_daily_analysis()