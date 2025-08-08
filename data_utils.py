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
import sanpy # Import the official sanpy library
from lunarcrush import LunarCrush # Import the official LunarCrush library

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
        print(f"   [WARN] Could not fetch CoinGlass data: {e}")
        return data

def fetch_santiment_data(slug: str) -> dict:
    print(f"   [INFO] Fetching on-chain/social data for {slug} from Santiment...")
    api_key = os.getenv("SANTIMENT_API_KEY")
    if not api_key:
        print("   [WARN] SANTIMENT_API_KEY not found. Skipping.")
        return {}
    
    sanpy.ApiConfig.api_key = api_key
    
    try:
        mvrv_data = sanpy.get(f"mvrv_usd/{slug}", from_date="utc_now-2d", to_date="utc_now", interval="1d")
        social_dom_data = sanpy.get(f"social_dominance_total/{slug}", from_date="utc_now-2d", to_date="utc_now", interval="1d")
        daa_data = sanpy.get(f"daily_active_addresses/{slug}", from_date="utc_now-2d", to_date="utc_now", interval="1d")

        metrics = {
            'mvrv_usd': mvrv_data.iloc[-1]['value'] if not mvrv_data.empty else 0.0,
            'social_dominance': social_dom_data.iloc[-1]['value'] if not social_dom_data.empty else 0.0,
            'daily_active_addresses': daa_data.iloc[-1]['value'] if not daa_data.empty else 0.0
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
    try:
        client = LunarCrush(api_key)
        assets = client.get_assets(symbol=api_symbol)
        data = assets.get('data', [{}])[0]

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
    print(f"   [INFO] Fetching CoinGecko data for {coin_id}...")
    api_key = os.getenv("COINGECKO_API_KEY")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {'x_cg_demo_api_key': api_key}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        metrics = {
            'market_cap_rank': data.get('market_cap_rank', 0),
            'ath_usd': data.get('market_data', {}).get('ath', {}).get('usd', 0)
        }
        print("   [SUCCESS] CoinGecko data fetched.")
        return metrics
    except requests.exceptions.RequestException as e:
        print(f"   [WARN] Could not fetch CoinGecko data for {coin_id}: {e}")
        return {}

def fetch_data(coin: str) -> pd.DataFrame:
    coingecko_map = {"BTC-USD": "bitcoin", "ETH-USD": "ethereum", "XRP-USD": "ripple"}
    santiment_slug = coingecko_map.get(coin)

    print(f"   [INFO] Fetching 180 days of historical data for {coin}...")
    try:
        df = yf.download(tickers=coin, period="180d", interval="1d", progress=False, auto_adjust=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # Technical Indicators
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

        # Integrate ALL Advanced Data Sources
        cg_data = fetch_coingecko_data(santiment_slug)
        futures_data = fetch_coinglass_data(coin)
        santiment_data = fetch_santiment_data(santiment_slug)
        lunar_data = fetch_lunarcrush_data(coin)
        
        # Add all data points to the DataFrame, ensuring fallbacks are numeric
        df['Market_Cap_Rank'] = cg_data.get('market_cap_rank', 0)
        df['All_Time_High_Real'] = cg_data.get('ath_usd', 0.0)
        df['Funding_Rate'] = futures_data.get('funding_rate', 0.0)
        df['Open_Interest'] = futures_data.get('open_interest', 0.0)
        df['Long_Short_Ratio'] = futures_data.get('long_short_ratio', 0.0)
        df['MVRV_Ratio'] = santiment_data.get('mvrv_usd', 0.0)
        df['Social_Dominance'] = santiment_data.get('social_dominance', 0.0)
        df['Daily_Active_Addresses'] = santiment_data.get('daily_active_addresses', 0.0)
        df['Galaxy_Score'] = lunar_data.get('galaxy_score', 0.0)
        df['Alt_Rank'] = lunar_data.get('alt_rank', 0)
        df['Exchange_Net_Flow'] = 0.0 # Placeholder as we don't have a free source for this yet

        df.dropna(inplace=True)
        print(f"   [SUCCESS] Data processing complete for {coin}.")
        return df
    except Exception as e:
        print(f"   [ERROR] An error occurred in fetch_data for {coin}: {e}")
        return pd.DataFrame()
