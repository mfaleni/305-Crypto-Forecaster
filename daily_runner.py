import pandas as pd
from datetime import datetime
import numpy as np
import os
import json
import openai
from frozendict import frozendict
from dotenv import load_dotenv

# --- Robust Local Path Configuration ---
# Get the absolute path of the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define file paths relative to the script's location
RESULTS_FILE = os.path.join(SCRIPT_DIR, 'forecast_results.csv')
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
# ------------------------------------


# --- Load and set API Keys ---
load_dotenv()
openai.api_key = os.getenv("REMOVED_OPENAI_KEY")
news_api_key = os.getenv("REMOVED_NEWSAPI_KEY")

if not openai.api_key or not news_api_key:
    print("❌ [FATAL] OpenAI or NewsAPI key not found. Please set them in your .env file or environment variables.")
    exit()

# --- Module Imports ---
try:
    from data_utils import fetch_data
    from forecasting import prophet_forecast, lstm_forecast, prophet_forecast_highs
    from sentiment import get_news_sentiment
except ImportError as e:
    print(f"❌ [FATAL] Failed to import a required module: {e}. Exiting.")
    exit()

# --- Configuration ---
COINS = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "XRP-USD": "XRP"}
RESULTS_FILE = os.path.join(MOUNT_PATH, 'forecast_results.csv')
DATA_DIR = os.path.join(MOUNT_PATH, 'data')

# Helper function to convert non-serializable objects
def default_json_serializer(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, frozendict):
        return dict(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def run_daily_analysis():
    """
    Orchestrates the daily analysis with robust error handling and correct file paths.
    """
    print("✅ [START] Kicking off daily crypto forecasting run...")
    today = datetime.today().strftime("%Y-%m-%d")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    all_results = []

    for ticker, name in COINS.items():
        print(f"\nProcessing {ticker} ({name})...")
        
        try:
            market_data = fetch_data(ticker)
            
            if market_data.empty or len(market_data) < 61:
                print(f"   [WARN] Insufficient or no data returned for {ticker}. Skipping this coin.")
                continue

            detailed_data_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
            market_data.to_csv(detailed_data_path)
            print(f"   [INFO] Detailed data saved to {detailed_data_path}")

            all_time_high = market_data['High'].max()
            actual_price = market_data["Close"].iloc[-1]
            prophet_price = prophet_forecast(market_data.copy())
            lstm_price = lstm_forecast(market_data.copy())
            sentiment_score = get_news_sentiment(coin_ticker=ticker, coin_name=name, api_key=news_api_key)
            high_forecasts_list = prophet_forecast_highs(market_data.copy(), periods=5)
            
            result = {
                "Date": today, "Coin": ticker, "Actual_Price": actual_price,
                "Prophet_Forecast": prophet_price, "LSTM_Forecast": lstm_price,
                "Sentiment_Score": sentiment_score, "All_Time_High": all_time_high,
                "High_Forecast_5_Day": json.dumps(high_forecasts_list, default=default_json_serializer)
            }
            all_results.append(result)

            print("\n--- 📈 Daily Report ---")
            print(f"Coin              : {ticker}")
            print(f"Actual Price      : ${actual_price:,.2f}")
            print(f"All-Time High     : ${all_time_high:,.2f}")
            print(f"Prophet Forecast  : ${prophet_price:,.2f}")
            print(f"LSTM Forecast     : ${lstm_price:,.2f}")
            print(f"Sentiment Score   : {sentiment_score:.2f}")
            print("-------------------------\n")

        except Exception as e:
            print(f"❌ [ERROR] An unexpected error occurred while processing {ticker}: {e}")
            continue

    print("\n✅ [FINISH] Daily crypto forecasting run complete.")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.fillna('N/A', inplace=True)
        results_df.to_csv(RESULTS_FILE, index=False)
        print(f"\n[INFO] Summary results saved to {RESULTS_FILE}")
    else:
        print("\n[WARN] No results were generated. Summary file not updated.")

if __name__ == "__main__":
    run_daily_analysis()