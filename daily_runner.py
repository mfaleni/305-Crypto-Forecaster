import pandas as pd
from datetime import datetime
import numpy as np
import os
import json
import openai
from frozendict import frozendict
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
news_api_key = os.getenv("NEWS_API_KEY")

if not openai.api_key or not news_api_key:
    print("❌ [FATAL] OPENAI_API_KEY or NEWS_API_KEY not found.")
    exit(1)

try:
    from data_utils import fetch_data
    from forecasting import prophet_forecast, lstm_forecast, prophet_forecast_highs
    from sentiment import get_news_sentiment
    from db_utils import init_db, save_forecast_results
    from analyst import get_daily_analysis
except ImportError as e:
    print(f"❌ [FATAL] Failed to import a required module: {e}. Exiting.")
    exit(1)

COINS = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "XRP-USD": "XRP"}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def default_json_serializer(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, frozendict):
        return dict(obj)
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
            
            all_time_high = market_data['High'].max()
            actual_price = market_data["Close"].iloc[-1]
            prophet_price = prophet_forecast(market_data.copy())
            lstm_price = lstm_forecast(market_data.copy())
            high_forecasts_list = prophet_forecast_highs(market_data.copy(), periods=5)
            latest_rsi = market_data["RSI"].iloc[-1]
            latest_macd = market_data["MACD"].iloc[-1]
            sentiment_score, top_headlines = get_news_sentiment(coin_ticker=ticker, coin_name=name, api_key=news_api_key)

            daily_briefing_data = {
                "coin_name": name, "actual_price": actual_price, "prophet_forecast": prophet_price,
                "sentiment_score": sentiment_score, "rsi": latest_rsi, "macd": latest_macd,
                "top_headlines": top_headlines
            }
            analysis_results = get_daily_analysis(daily_briefing_data)

            result = {
                "Date": today, "Coin": ticker, "Actual_Price": actual_price,
                "Prophet_Forecast": prophet_price, "LSTM_Forecast": lstm_price,
                "Sentiment_Score": sentiment_score, "RSI": latest_rsi, "MACD": latest_macd,
                "All_Time_High": all_time_high,
                "High_Forecast_5_Day": json.dumps(high_forecasts_list, default=default_json_serializer),
                "analysis_summary": analysis_results.get("summary"),
                "analysis_hypothesis": analysis_results.get("hypothesis"),
                "analysis_news_links": analysis_results.get("news_links")
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