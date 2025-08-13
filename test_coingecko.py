import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
COIN_ID = "bitcoin" # We'll test with bitcoin

if not COINGECKO_API_KEY:
    print("❌ COINGECKO_API_KEY not found in .env file. Exiting.")
    exit()

print(f"--- Definitive CoinGecko API Test for {COIN_ID} ---")
print(f"Using Key: ...{COINGECKO_API_KEY[-4:]}")

url = f"https://api.coingecko.com/api/v3/coins/{COIN_ID}"
params = {'x_cg_demo_api_key': COINGECKO_API_KEY}

try:
    print("\n--- Testing Endpoint: CoinGecko Coins ---")
    response = requests.get(url, params=params)
    
    print(f"Status Code: {response.status_code}")

    if response.ok:
        json_data = response.json()
        print("\n--- Parsing Response ---")

        community_score = json_data.get('community_score', "Not Available")
        developer_score = json_data.get('developer_score', "Not Available")

        print("\n--- Extracted Metrics ---")
        print(f"Community Score: {community_score}")
        print(f"Developer Score: {developer_score}")

        if community_score != "Not Available" and developer_score != "Not Available":
            print("\n🎉 SUCCESS: All key metrics were extracted successfully.")
        else:
            print("\n⚠️ WARNING: One or more metrics were not available.")
            print("Full Response (first 500 chars):", str(json_data)[:500])
    else:
        print(f"❌ ERROR: API call failed.")
        print("Response Text:", response.text)

except Exception as e:
    print(f"An error occurred: {e}")