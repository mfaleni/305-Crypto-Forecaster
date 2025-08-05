import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

def prophet_forecast(df: pd.DataFrame) -> float:
    if df.empty: return np.nan
    print("   [INFO] Starting Prophet 'Close' price forecast...")
    try:
        prophet_df = pd.DataFrame({'ds': df.index, 'y': df['Close'].values})
        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)
        predicted_price = forecast.iloc[-1]['yhat']
        print(f"   [SUCCESS] Prophet 'Close' forecast complete. Predicted: {predicted_price:.2f}")
        return float(predicted_price)
    except Exception as e:
        print(f"   [ERROR] Prophet 'Close' forecasting error: {e}")
        return np.nan

def prophet_forecast_highs(df: pd.DataFrame, periods: int = 5) -> list:
    if df.empty: return []
    print(f"   [INFO] Starting Prophet {periods}-day 'High' price forecast...")
    try:
        prophet_df = pd.DataFrame({'ds': df.index, 'y': df['High'].values})
        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        future_forecasts = forecast.iloc[-periods:]
        predicted_highs = future_forecasts[['ds', 'yhat']].to_dict('records')
        print(f"   [SUCCESS] Prophet {periods}-day 'High' forecast complete.")
        return predicted_highs
    except Exception as e:
        print(f"   [ERROR] Prophet 'High' forecasting error: {e}")
        return []

def lstm_forecast(df: pd.DataFrame, look_back_period: int = 60) -> float:
    if df.empty or len(df) <= look_back_period: return np.nan
    print("   [INFO] Starting LSTM 'Close' price forecast...")
    try:
        data = df[['Close']].copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        X_train, y_train = [], []
        for i in range(look_back_period, len(scaled_data)):
            X_train.append(scaled_data[i-look_back_period:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=0)
        last_sequence = scaled_data[-look_back_period:]
        last_sequence = np.reshape(last_sequence, (1, look_back_period, 1))
        predicted_price_scaled = model.predict(last_sequence, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        print(f"   [SUCCESS] LSTM 'Close' forecast complete. Predicted: {predicted_price[0][0]:,.2f}")
        return float(predicted_price[0][0])
    except Exception as e:
        print(f"   [ERROR] LSTM forecasting error: {e}")
        return np.nan