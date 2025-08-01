import pandas as pd
from datetime import datetime
import numpy as np
import os
import json
import openai
from frozendict import frozendict

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

openai.api_key = os.getenv("REMOVED_OPENAI_KEY")
news_api_key = os.getenv("REMOVED_NEWSAPI_KEY")

# Import project modules
from data_utils import fetch_data
from forecasting import prophet_forecast, lstm_forecast, prophet_forecast_highs
from sentiment import get_news_sentiment

# Configuration
COINS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "XRP-USD": "XRP",
}
RESULTS_FILE = 'forecast_results.csv'
DATA_DIR = 'data' # Directory to store detailed data files

# Helper function to convert non-serializable objects to a serializable format
def default_json_serializer(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, frozendict):
        return dict(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def run_daily_analysis():
    """
    Orchestrates the daily analysis, now including a 5-day 'High' forecast.
    """
    print("✅ [START] Kicking off daily crypto forecasting run...")
    today = datetime.today().strftime("%Y-%m-%d")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    all_results = []

    for ticker, name in COINS.items():
        print(f"\nProcessing {ticker} ({name})...")
        
        try:
            # 1. Fetch comprehensive data
            market_data = fetch_data(ticker)
            if market_data.empty or len(market_data) < 61:
                print(f"   [WARN] Insufficient data for {ticker}. Skipping.")
                continue
            
            # 2. Save the detailed data for the dashboard
            detailed_data_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
            market_data.to_csv(detailed_data_path)
            print(f"   [INFO] Detailed data saved to {detailed_data_path}")

            # 3. Calculate All-Time High (ATH) from the available data
            all_time_high = market_data['High'].max()
            
            # 4. Run forecasts and sentiment analysis
            actual_price = market_data["Close"].iloc[-1].item()
            prophet_price = prophet_forecast(market_data.copy())
            lstm_price = lstm_forecast(market_data.copy())
            sentiment_score = get_news_sentiment(coin_ticker=ticker, coin_name=name)
            
            # 5. Get the new 5-day 'High' forecast and convert to serializable format
            # prophet_forecast_highs now returns a list, not a DataFrame
            high_forecasts_list = prophet_forecast_highs(market_data.copy(), periods=5)
            
            # 6. Store summary result, now including ATH and 5-day forecast
            result = {
                "Date": today, "Coin": ticker, "Actual_Price": actual_price,
                "Prophet_Forecast": prophet_price if not np.isnan(prophet_price) else 0.0,
                "LSTM_Forecast": lstm_price if not np.isnan(lstm_price) else 0.0,
                "Sentiment_Score": sentiment_score,
                "All_Time_High": all_time_high,
                # Store the 5-day forecast as a JSON string in the CSV
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
            print(f"❌ [ERROR] An error occurred while processing {ticker}: {e}")
            continue

    print("\n✅ [FINISH] Daily crypto forecasting run complete.")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_FILE, index=False)
        print(f"\n[INFO] Summary results saved to {RESULTS_FILE}")
    else:
        print("\n[WARN] No results were generated. Summary file not updated.")

if __name__ == "__main__":
    run_daily_analysis()