import pandas as pd
from datetime import datetime
import numpy as np
import os
import json
import openai
from frozendict import frozendict
from dotenv import load_dotenv

# --- Load Environment and Keys ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

# --- Robust Key Check ---
required_keys = [
    "OPENAI_API_KEY", "NEWS_API_KEY", "SANTIMENT_API_KEY", 
    "LUNARCRUSH_API_KEY", "COINGECKO_API_KEY"
]
missing_keys = [key for key in required_keys if not os.getenv(key)]
if missing_keys:
    print(f"❌ [FATAL] The following required API keys are missing: {', '.join(missing_keys)}")
    exit(1)

# --- Module Imports ---
try:
    from data_utils import fetch_data
    from forecasting import prophet_forecast, lstm_forecast, prophet_forecast_highs
    from sentiment import get_news_sentiment
    from db_utils import init_db, save_forecast_results
    from analyst import get_daily_analysis
except ImportError as e:
    print(f"❌ [FATAL] Failed to import a required module: {e}. Exiting.")
    exit(1)

# --- Configuration ---
COINS = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "XRP-USD": "XRP"}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def default_json_serializer(obj):
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    if isinstance(obj, frozendict): return dict(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def run_daily_analysis():
    print("✅ [START] Kicking off daily crypto forecasting run...")
    init_db()
    today = datetime.today().strftime("%Y-%m-%d")
    os.makedirs(DATA_DIR, exist_ok=True)
    all_results = []
    for ticker, name in COINS.items():
        print(f"\nProcessing {ticker} ({name})...")
        try:
            market_data = fetch_data(ticker)
            if market_data.empty or len(market_data) < 61:
                print(f"   [WARN] Insufficient data for {ticker}. Skipping.")
                continue
            
            detailed_data_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
            market_data.to_csv(detailed_data_path)
            
            latest_data = market_data.iloc[-1]
            
            prophet_price = prophet_forecast(market_data.copy())
            lstm_price = lstm_forecast(market_data.copy())
            high_forecasts_list = prophet_forecast_highs(market_data.copy(), periods=5)
            sentiment_score, top_headlines = get_news_sentiment(coin_ticker=ticker, coin_name=name, api_key=news_api_key)

            # Prepare a comprehensive briefing for the AI Analyst
            daily_briefing_data = {
                "coin_name": name,
                "actual_price": latest_data.get("Close", 0.0),
                "prophet_forecast": prophet_price,
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
                # START: ADDED NEW DATA TO BRIEFING
                "leverage_ratio": latest_data.get("Leverage_Ratio", 0.0),
                "futures_volume_24h": latest_data.get("Futures_Volume_24h", 0.0),
                "exchange_supply_ratio": latest_data.get("Exchange_Supply_Ratio", 0.0),
                # END: ADDED NEW DATA TO BRIEFING
                "top_headlines": top_headlines
            }
            analysis_results = get_daily_analysis(daily_briefing_data)

            # Assemble the final, complete record for the database
            result = {
                "Date": today,
                "Coin": ticker,
                "Actual_Price": latest_data.get("Close", 0.0),
                "Prophet_Forecast": prophet_price,
                "LSTM_Forecast": lstm_price,
                "Sentiment_Score": sentiment_score,
                "RSI": latest_data.get("RSI", 0.0),
                "MACD": latest_data.get("MACD", 0.0),
                "All_Time_High": latest_data.get("All_Time_High_Real", 0.0),
                "High_Forecast_5_Day": json.dumps(high_forecasts_list, default=default_json_serializer),
                "Funding_Rate": latest_data.get("Funding_Rate", 0.0),
                "Open_Interest": latest_data.get("Open_Interest", 0.0),
                "Long_Short_Ratio": latest_data.get("Long_Short_Ratio", 0.0),
                "MVRV_Ratio": latest_data.get("MVRV_Ratio", 0.0),
                "Social_Dominance": latest_data.get("Social_Dominance", 0.0),
                "Daily_Active_Addresses": latest_data.get("Daily_Active_Addresses", 0.0),
                "Galaxy_Score": latest_data.get("Galaxy_Score", 0.0),
                "Alt_Rank": latest_data.get("Alt_Rank", 0.0),
                "Exchange_Net_Flow": latest_data.get("Exchange_Net_Flow", 0.0),
                # START: ADDED NEW DATA TO SAVE
                "Leverage_Ratio": latest_data.get("Leverage_Ratio", 0.0),
                "Futures_Volume_24h": latest_data.get("Futures_Volume_24h", 0.0),
                "Exchange_Supply_Ratio": latest_data.get("Exchange_Supply_Ratio", 0.0),
                # END: ADDED NEW DATA TO SAVE
                "analysis_summary": analysis_results.get("summary"),
                "analysis_hypothesis": analysis_results.get("hypothesis"),
                "analysis_news_links": analysis_results.get("news_links"),
                "report_title": analysis_results.get("report_title"),
                "report_recap": analysis_results.get("report_recap"),
                "report_bullish": analysis_results.get("report_bullish"),
                "report_bearish": analysis_results.get("report_bearish"),
                "report_hypothesis": analysis_results.get("report_hypothesis"),
                "user_feedback": None,
                "user_correction": None
            }
            all_results.append(result)

        except Exception as e:
            print(f"❌ [ERROR] An unexpected error occurred while processing {ticker}: {e}")
            continue
            
    print("\n✅ [FINISH] Daily processing complete.")
    if all_results:
        results_df = pd.DataFrame(all_results)
        save_forecast_results(results_df)
    else:
        print("\n[WARN] No results were generated. Database not updated.")

if __name__ == "__main__":
    run_daily_analysis()