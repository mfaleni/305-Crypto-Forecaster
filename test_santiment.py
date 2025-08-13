import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SANTIMENT_API_KEY = os.getenv("SANTIMENT_API_KEY")
SLUG = "bitcoin" # We'll test with bitcoin

if not SANTIMENT_API_KEY:
    print("❌ SANTIMENT_API_KEY not found in .env file. Exiting.")
    exit()

print(f"--- Definitive Santiment API Test for {SLUG} ---")
print(f"Using Key: ...{SANTIMENT_API_KEY[-4:]}")

# This is the GraphQL query to get the metrics we need
query = f"""
query {{
  mvrv: getMetric(metric: "mvrv_usd") {{
    timeseriesData(slug: "{SLUG}", from: "utc_now-2d", to: "utc_now", interval: "1d") {{ value }}
  }}
  social_dominance: getMetric(metric: "social_dominance_total") {{
    timeseriesData(slug: "{SLUG}", from: "utc_now-2d", to: "utc_now", interval: "1d") {{ value }}
  }}
}}
"""

try:
    print("\n--- Testing Endpoint: Santiment GraphQL ---")
    response = requests.post('https://api.santiment.net/graphql', 
                             json={'query': query}, 
                             headers={'Authorization': f'Apikey {SANTIMENT_API_KEY}'})
    
    print(f"Status Code: {response.status_code}")

    if response.ok:
        json_data = response.json()
        print("\n--- Parsing Response ---")

        if json_data.get('data'):
            mvrv_data = json_data['data'].get('mvrv', {}).get('timeseriesData', [])
            social_data = json_data['data'].get('social_dominance', {}).get('timeseriesData', [])

            mvrv_value = mvrv_data[-1]['value'] if mvrv_data and mvrv_data[-1] else "Not Available"
            social_value = social_data[-1]['value'] if social_data and social_data[-1] else "Not Available"

            print("\n--- Extracted Metrics ---")
            print(f"MVRV Ratio: {mvrv_value}")
            print(f"Social Dominance: {social_value}")

            if mvrv_value != "Not Available" and social_value != "Not Available":
                print("\n🎉 SUCCESS: All key metrics were extracted successfully.")
            else:
                print("\n⚠️ WARNING: One or more metrics were not available. This is likely a Santiment plan limitation.")
                print("Full Response:", json.dumps(json_data, indent=2))
        else:
            print("❌ ERROR: API response did not contain a 'data' key.")
            print("Full Response:", json.dumps(json_data, indent=2))
    else:
        print(f"❌ ERROR: API call failed.")
        print("Response Text:", response.text)

except Exception as e:
    print(f"An error occurred: {e}")