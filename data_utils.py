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
        # Download historical data
        df = yf.download(
            tickers=coin, period="180d", interval="1d", progress=False, auto_adjust=False
        )

        if df.empty:
            print(f"   [WARN] No data found for {coin}.")
            return pd.DataFrame()

        # --- FIX for MultiIndex columns ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # --- ✅ 1. Technical Indicators ---
        print("   [INFO] Calculating technical indicators...")
        # Simple Moving Average (SMA)
        df['SMA'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        # Exponential Moving Average (EMA)
        df['EMA'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        # Relative Strength Index (RSI)
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()
        # Moving Average Convergence Divergence (MACD)
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        # Bollinger Bands
        bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['Stoch_k'] = stoch.stoch()
        df['Stoch_d'] = stoch.stoch_signal()
        # On-Balance Volume (OBV)
        df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        # Ichimoku Cloud
        ichimoku = IchimokuIndicator(high=df['High'], low=df['Low'])
        df['Ichimoku_a'] = ichimoku.ichimoku_a()
        df['Ichimoku_b'] = ichimoku.ichimoku_b()

        # --- ✅ 2. On-Chain Indicators (Simulated) ---
        print("   [INFO] Simulating on-chain indicators...")
        df['Active_Addresses'] = np.random.randint(100000, 1000000, size=len(df))
        df['Transaction_Volume'] = np.random.uniform(1e9, 5e10, size=len(df))
        df['Supply_Distribution'] = np.random.rand(len(df))
        df['TVL'] = np.random.uniform(1e9, 5e11, size=len(df))
        df['Realized_PnL'] = np.random.uniform(-1e8, 1e8, size=len(df))

        # --- ✅ 3. Fundamental Indicators (Simulated) ---
        print("   [INFO] Simulating fundamental indicators...")
        df['Token_Utility'] = np.random.uniform(1, 10, size=len(df))
        df['Adoption_Rate'] = np.random.uniform(1, 10, size=len(df))
        df['Team_Score'] = np.random.uniform(1, 10, size=len(df))
        df['Tokenomics_Score'] = np.random.uniform(1, 10, size=len(df))
        df['Regulatory_Risk'] = np.random.uniform(1, 10, size=len(df))

        # --- Final Data Cleaning ---
        df.dropna(inplace=True)

        print(f"   [SUCCESS] Data processed for {coin}. Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"   [ERROR] An error occurred while processing data for {coin}: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # This block allows for direct testing of this module
    print("--- Testing data_utils.py ---")
    
    btc_data = fetch_data("BTC-USD")
    if not btc_data.empty:
        print("\nBTC-USD Data Head (Sample of new columns):")
        print(btc_data[['Close', 'SMA', 'RSI', 'BB_High', 'Active_Addresses', 'Team_Score']].head())
        print(f"\nTotal columns: {len(btc_data.columns)}")
