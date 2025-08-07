import pandas as pd
import yfinance as yf
import requests
import os
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, IchimokuIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import numpy as np
from datetime import datetime, timedelta

# --- API HELPER FUNCTIONS ---

def fetch_coinglass_data(symbol: str) -> dict:
    print(f"   [INFO] Fetching futures data for {symbol} from CoinGlass...")
    headers = {'accept': 'application/json'}
    api_symbol = symbol.replace("-USD", "")
    data = {'funding_rate': 0.0, 'open_interest': 0.0, 'long_short_ratio': 0.0}
    try:
        funding_url = f"https://open-api.coinglass.com/public/v2/funding?ex=Binance&symbol={api_symbol}"
        oi_url = f"https://open-api.coinglass.com/public/v2/open_interest?ex=Binance&symbol={api_symbol}"
        ls_url = f"https://open-api.coinglass.com/public/v2/long_short?ex=Binance&symbol={api_symbol}"

        funding_res = requests.get(funding_url, headers=headers)
        if funding_res.ok and funding_res.json().get('data'):
            data['funding_rate'] = funding_res.json()['data'][0].get('rate', 0.0) * 100
        
        oi_res = requests.get(oi_url, headers=headers)
        if oi_res.ok and oi_res.json().get('data'):
            data['open_interest'] = oi_res.json()['data'][0].get('openInterest', 0.0)

        ls_res = requests.get(ls_url, headers=headers)
        if ls_res.ok and ls_res.json().get('data'):
            data['long_short_ratio'] = ls_res.json()['data'][0].get('longShortRatio', 0.0)
        
        print("   [SUCCESS] Futures data fetched.")
        return data
    except Exception as e:
        print(f"   [WARN] Could not fetch futures data: {e}")
        return data

def fetch_santiment_data(slug: str) -> dict:
    print(f"   [INFO] Fetching on-chain/social data for {slug} from Santiment...")
    api_key = os.getenv("SANTIMENT_API_KEY")
    if not api_key:
        print("   [WARN] SANTIMENT_API_KEY not found. Skipping.")
        return {}
    
    # GraphQL query to get multiple metrics in one call
    query = f"""
    {{
      getMetric(metric: "mvrv_usd") {{
        timeseriesData(slug: "{slug}", from: "utc_now-1d", to: "utc_now", interval: "1d") {{ value }}
      }}
      getMetric(metric: "social_dominance_total") {{
        timeseriesData(slug: "{slug}", from: "utc_now-1d", to: "utc_now", interval: "1d") {{ value }}
      }}
      getMetric(metric: "daily_active_addresses") {{
        timeseriesData(slug: "{slug}", from: "utc_now-1d", to: "utc_now", interval: "1d") {{ value }}
      }}
    }}
    """
    try:
        response = requests.post('https://api.santiment.net/graphql', json={'query': query}, headers={'Authorization': f'Apikey {api_key}'})
        response.raise_for_status()
        data = response.json().get('data', {})
        
        metrics = {
            'mvrv_usd': data.get('getMetric', {}).get('timeseriesData', [{}])[0].get('value', 0.0),
            'social_dominance': data.get('getMetric', {}).get('timeseriesData', [{}])[0].get('value', 0.0),
            'daily_active_addresses': data.get('getMetric', {}).get('timeseriesData', [{}])[0].get('value', 0.0)
        }
        print("   [SUCCESS] Santiment data fetched.")
        return metrics
    except Exception as e:
        print(f"   [WARN] Could not fetch Santiment data: {e}")
        return {}

def fetch_lunarcrush_data(symbol: str) -> dict:
    print(f"   [INFO] Fetching social intelligence for {symbol} from LunarCrush...")
    api_key = os.getenv("LUNARCRUSH_API_KEY")
    if not api_key:
        print("   [WARN] LUNARCRUSH_API_KEY not found. Skipping.")
        return {}
    
    api_symbol = symbol.replace("-USD", "")
    url = f"https://lunarcrush.com/api3/coins/{api_symbol}/meta"
    headers = {'Authorization': f'Bearer {api_key}'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json().get('data', {})
        
        metrics = {
            'galaxy_score': data.get('galaxy_score', 0.0),
            'alt_rank': data.get('alt_rank', 0)
        }
        print("   [SUCCESS] LunarCrush data fetched.")
        return metrics
    except Exception as e:
        print(f"   [WARN] Could not fetch LunarCrush data: {e}")
        return {}

def fetch_coingecko_data(coin_id: str) -> dict:
    # This function remains unchanged
    pass

def fetch_data(coin: str) -> pd.DataFrame:
    """
    Fetches historical data, calculates technical indicators, and enriches
    it with data from all integrated professional sources.
    """
    coingecko_map = {"BTC-USD": "bitcoin", "ETH-USD": "ethereum", "XRP-USD": "ripple"}
    santiment_slug = coingecko_map.get(coin)

    print(f"   [INFO] Fetching 180 days of historical data for {coin}...")
    try:
        df = yf.download(tickers=coin, period="180d", interval="1d", progress=False, auto_adjust=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # Technical Indicators (Unchanged)
        print("   [INFO] Calculating technical indicators...")
        df['SMA'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['EMA'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd(); df['MACD_Signal'] = macd.macd_signal()
        bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High'] = bollinger.bollinger_hband(); df['BB_Low'] = bollinger.bollinger_lband()
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['Stoch_k'] = stoch.stoch(); df['Stoch_d'] = stoch.stoch_signal()
        df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'])
        df['Ichimoku_a'] = ichimoku.ichimoku_a(); df['Ichimoku_b'] = ichimoku.ichimoku_b()

        # --- Integrate ALL Advanced Data Sources ---
        cg_data = fetch_coingecko_data(santiment_slug)
        futures_data = fetch_coinglass_data(coin)
        santiment_data = fetch_santiment_data(santiment_slug)
        lunar_data = fetch_lunarcrush_data(coin)
        
        # Add all data points to the DataFrame
        if cg_data:
            df['Market_Cap_Rank'] = cg_data.get('market_cap_rank')
            df['All_Time_High_Real'] = cg_data.get('ath_usd')
        if futures_data:
            df['Funding_Rate'] = futures_data.get('funding_rate')
            df['Open_Interest'] = futures_data.get('open_interest')
            df['Long_Short_Ratio'] = futures_data.get('long_short_ratio')
        if santiment_data:
            df['MVRV_Ratio'] = santiment_data.get('mvrv_usd')
            df['Social_Dominance'] = santiment_data.get('social_dominance')
            df['Daily_Active_Addresses'] = santiment_data.get('daily_active_addresses')
        if lunar_data:
            df['Galaxy_Score'] = lunar_data.get('galaxy_score')
            df['Alt_Rank'] = lunar_data.get('alt_rank')
        
        df.dropna(inplace=True)
        print(f"   [SUCCESS] Data processing complete for {coin}.")
        return df
    except Exception as e:
        print(f"   [ERROR] An error occurred in fetch_data for {coin}: {e}")
        return pd.DataFrame()
