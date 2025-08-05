import pandas as pd
import yfinance as yf
import requests
from ta.momentum import RSIIndicator, StochasticOscillator
# --- THIS LINE IS NOW CORRECTED ---
from ta.trend import MACD, SMAIndicator, EMAIndicator, IchimokuIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import numpy as np

def fetch_coingecko_data(coin_id: str) -> dict:
    """Fetches fundamental and market data from the CoinGecko API."""
    print(f"   [INFO] Fetching CoinGecko data for {coin_id}...")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        metrics = {
            'market_cap_rank': data.get('market_cap_rank', 0),
            'community_score': data.get('community_score', 0),
            'developer_score': data.get('developer_score', 0),
            'sentiment_up_percentage': data.get('sentiment_votes_up_percentage', 0),
            'total_volume': data.get('market_data', {}).get('total_volume', {}).get('usd', 0),
            'circulating_supply': data.get('market_data', {}).get('circulating_supply', 0),
            'ath_usd': data.get('market_data', {}).get('ath', {}).get('usd', 0)
        }
        print("   [SUCCESS] CoinGecko data fetched.")
        return metrics
    except requests.exceptions.RequestException as e:
        print(f"   [WARN] Could not fetch CoinGecko data for {coin_id}: {e}")
        return {}

def fetch_data(coin: str) -> pd.DataFrame:
    """
    Fetches historical data, calculates technical indicators, and enriches
    it with real data from CoinGecko.
    """
    coingecko_map = {
        "BTC-USD": "bitcoin", "ETH-USD": "ethereum", "XRP-USD": "ripple",
    }
    coin_id = coingecko_map.get(coin)

    print(f"   [INFO] Fetching 180 days of historical data for {coin}...")
    try:
        df = yf.download(tickers=coin, period="180d", interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            print(f"   [WARN] No data found for {coin}.")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        print("   [INFO] Calculating technical indicators...")
        df['SMA'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['EMA'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['Stoch_k'] = stoch.stoch()
        df['Stoch_d'] = stoch.stoch_signal()
        df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'])
        df['Ichimoku_a'] = ichimoku.ichimoku_a()
        df['Ichimoku_b'] = ichimoku.ichimoku_b()

        if coin_id:
            cg_data = fetch_coingecko_data(coin_id)
            if cg_data:
                print("   [INFO] Integrating real on-chain and fundamental data...")
                df['Transaction_Volume_24h'] = cg_data.get('total_volume')
                df['Circulating_Supply'] = cg_data.get('circulating_supply')
                df['Market_Cap_Rank'] = cg_data.get('market_cap_rank')
                df['Community_Score'] = cg_data.get('community_score')
                df['Developer_Score'] = cg_data.get('developer_score')
                df['Sentiment_Up_Percentage'] = cg_data.get('sentiment_up_percentage')
                df['All_Time_High_Real'] = cg_data.get('ath_usd')
        
        df.dropna(inplace=True)
        print(f"   [SUCCESS] Data processed for {coin}.")
        return df

    except Exception as e:
        print(f"   [ERROR] An error occurred while processing data for {coin}: {e}")
        return pd.DataFrame()