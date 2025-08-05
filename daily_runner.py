import pandas as pd
from datetime import datetime
import numpy as np
import os
import json
from frozendict import frozendict
from dotenv import load_dotenv

# --- Load Environment and Keys ---
load_dotenv()
openai.api_key = os.getenv("REMOVED_OPENAI_KEY")
news_api_key = os.getenv("REMOVED_NEWSAPI_KEY")

if not openai.api_key or not news_api_key:
    print("❌ [FATAL] API keys not found.")
    exit()

# --- Module Imports ---
try:
    from data_utils import fetch_data
    from forecasting import prophet_forecast, lstm_forecast, prophet_forecast_highs
    from sentiment import get_news_sentiment
    from db_utils import init_db, save_forecast_results
except ImportError as e:
    print(f"❌ [FATAL] Failed to import a required module: {e}. Exiting.")
    exit()

# --- Configuration ---
COINS = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "XRP-USD": "XRP"}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data') # Still save detailed CSVs locally

# Helper function
def default_json_serializer(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, frozendict):
        return dict(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def run_daily_analysis():
    print("✅ [START] Kicking off daily crypto forecasting run...")
    # Initialize the database, create table if needed
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

            # We still save detailed daily data locally for the dashboard to read
            detailed_data_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
            market_data.to_csv(detailed_data_path)
            print(f"   [INFO] Detailed daily data saved to {detailed_data_path}")

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

        except Exception as e:
            print(f"❌ [ERROR] An unexpected error occurred while processing {ticker}: {e}")
            continue

    print("\n✅ [FINISH] Daily processing complete.")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        save_forecast_results(results_df) # Save to database
    else:
        print("\n[WARN] No results were generated. Database not updated.")

if __name__ == "__main__":
    run_daily_analysis()