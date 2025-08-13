import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY")
# --- MODIFIED: List of coins to test ---
COINS_TO_TEST = ["BTC", "ETH", "XRP"]

if not COINGLASS_API_KEY:
    print("❌ COINGLASS_API_KEY not found in .env file. Exiting.")
    exit()

print("--- Definitive CoinGlass API Multi-Coin Test ---")
print(f"Using Key: ...{COINGLASS_API_KEY[-4:]}")

# --- Loop through each coin ---
for symbol in COINS_TO_TEST:
    headers = {
        'accept': 'application/json',
        'coinglassSecret': COINGLASS_API_KEY 
    }
    url = f"https://open-api.coinglass.com/public/v2/perpetual_market?ex=Binance&symbol={symbol}"

    print(f"\n{'='*10} Testing: {symbol} {'='*10}")
    print(f"URL: {url}")
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")

        if response.ok:
            json_data = response.json()
            print("\n--- Parsing Response ---")

            if json_data.get('success') and json_data.get('data'):
                market_data_list = json_data['data'].get(symbol, [])
                binance_data = next((item for item in market_data_list if item.get("exchangeName") == "Binance"), None)

                if binance_data:
                    print("✅ Successfully found data for Binance exchange.")
                    
                    funding_rate = binance_data.get('rate', 0.0) * 100
                    open_interest = binance_data.get('openInterest', 0.0)
                    futures_volume = binance_data.get('totalVolUsd', 0.0)
                    
                    long_rate = binance_data.get('longRate', 0.0)
                    short_rate = binance_data.get('shortRate', 1.0)
                    long_short_ratio = long_rate / short_rate if short_rate > 0 else 0
                    
                    print("\n--- Extracted Metrics ---")
                    print(f"Funding Rate: {funding_rate:.4f}%")
                    print(f"Open Interest: ${open_interest:,.2f}")
                    print(f"Futures Volume (24h): ${futures_volume:,.2f}")
                    print(f"Long/Short Ratio: {long_short_ratio:.2f}")
                    
                    if funding_rate == 0 and open_interest == 0 and futures_volume == 0 and long_short_ratio == 0:
                         print("\n⚠️ WARNING: All extracted values are zero.")
                    else:
                        print("\n🎉 SUCCESS: All key metrics were extracted successfully.")

                else:
                    print(f"❌ ERROR: Could not find 'Binance' data for {symbol} within the API response.")

            else:
                print(f"❌ ERROR: API response for {symbol} indicates failure or is empty.")
                print("Full Response:", json.dumps(json_data, indent=2))

        else:
            print(f"❌ ERROR: API call for {symbol} failed.")
            print("Response Text:", response.text)

    except Exception as e:
        print(f"An error occurred for {symbol}: {e}")