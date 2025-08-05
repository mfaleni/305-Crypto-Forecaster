import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, IchimokuIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from ta.utils import dropna
import numpy as np

def fetch_data(coin: str) -> pd.DataFrame:
    """
    Fetches 180 days of historical cryptocurrency data and calculates a comprehensive
    set of technical, on-chain, and fundamental indicators.
    """
    print(f"   [INFO] Fetching 180 days of historical data for {coin}...")
    try:
        df = yf.download(
            tickers=coin, period="180d", interval="1d", progress=False, auto_adjust=False
        )
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

        print("   [INFO] Simulating on-chain indicators...")
        df['Active_Addresses'] = np.random.randint(100000, 1000000, size=len(df))
        df['Transaction_Volume'] = np.random.uniform(1e9, 5e10, size=len(df))
        df['TVL'] = np.random.uniform(1e9, 5e11, size=len(df))
        df['Realized_PnL'] = np.random.uniform(-1e8, 1e8, size=len(df))

        print("   [INFO] Simulating fundamental indicators...")
        df['Token_Utility'] = np.random.uniform(1, 10, size=len(df))
        df['Adoption_Rate'] = np.random.uniform(1, 10, size=len(df))
        df['Team_Score'] = np.random.uniform(1, 10, size=len(df))
        df['Tokenomics_Score'] = np.random.uniform(1, 10, size=len(df))
        df['Regulatory_Risk'] = np.random.uniform(1, 10, size=len(df))

        df.dropna(inplace=True)
        print(f"   [SUCCESS] Data processed for {coin}. Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"   [ERROR] An error occurred while processing data for {coin}: {e}")
        return pd.DataFrame()