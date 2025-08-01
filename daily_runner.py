import pandas as pd
from datetime import datetime
import numpy as np
import os

# --- Import project modules ---
# Use try-except blocks for robust importing, especially for deployment
try:
    from data_utils import fetch_data
    from forecasting import prophet_forecast, lstm_forecast
    from sentiment import get_news_sentiment
except ImportError:
    from .data_utils import fetch_data
    from .forecasting import prophet_forecast, lstm_forecast
    from .sentiment import get_news_sentiment


# --- Configuration ---
COINS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "XRP-USD": "XRP",
}
RESULTS_FILE = 'forecast_results.csv'

def run_daily_analysis():
    """
    The main function to orchestrate the daily forecasting and sentiment analysis process.
    """
    print("✅ [START] Kicking off daily crypto forecasting run...")
    today = datetime.today().strftime("%Y-%m-%d")
    
    all_results = []

    for ticker, name in COINS.items():
        print(f"\nProcessing {ticker} ({name})...")
        
        market_data = fetch_data(ticker)
        if market_data.empty or len(market_data) < 61:
            print(f"   [WARN] Insufficient data for {ticker}. Skipping.")
            continue
        
        actual_price = market_data["Close"].iloc[-1].item()
        prophet_price = prophet_forecast(market_data.copy())
        lstm_price = lstm_forecast(market_data.copy())
        sentiment_score = get_news_sentiment(coin_ticker=ticker, coin_name=name)
        
        result = {
            "Date": today,
            "Coin": ticker,
            "Actual_Price": actual_price,
            "Prophet_Forecast": prophet_price if not np.isnan(prophet_price) else 0.0,
            "LSTM_Forecast": lstm_price if not np.isnan(lstm_price) else 0.0,
            "Sentiment_Score": sentiment_score,
        }
        all_results.append(result)

        print("\n--- 📈 Daily Report ---")
        print(f"Coin              : {ticker}")
        print(f"Actual Price      : ${actual_price:,.2f}")
        print(f"Prophet Forecast  : ${prophet_price:,.2f}")
        print(f"LSTM Forecast     : ${lstm_price:,.2f}")
        print(f"Sentiment Score   : {sentiment_score:.2f}")
        print("-------------------------\n")

    print("\n✅ [FINISH] Daily crypto forecasting run complete.")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n--- 📋 Full Summary ---")
        print(results_df.to_string())
        results_df.to_csv(RESULTS_FILE, index=False)
        print(f"\n[INFO] Results saved to {RESULTS_FILE}")
    else:
        print("\n[WARN] No results were generated. CSV file not updated.")

# This block allows the script to be run directly from the terminal
if __name__ == "__main__":
    run_daily_analysis()
