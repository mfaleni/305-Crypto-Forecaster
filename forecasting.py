import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Import the data fetching function to use for direct testing
from data_utils import fetch_data

def prophet_forecast(df: pd.DataFrame) -> float:
    """
    Generates a next-day price forecast using the Prophet model.

    Args:
        df (pd.DataFrame): DataFrame containing historical data, including 'Close' prices.
                           The DataFrame's index should be a DatetimeIndex.

    Returns:
        float: The forecasted price for the next day. Returns np.nan on error.
    """
    if df.empty:
        print("   [WARN] Prophet forecast cannot run on an empty DataFrame.")
        return np.nan

    print("   [INFO] Starting Prophet forecast...")
    try:
        # Prophet requires a DataFrame with 'ds' (datestamp) and 'y' (value) columns.
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df.index
        prophet_df['y'] = df['Close'].values # Use .values to ensure it's a 1D array

        model = Prophet(daily_seasonality=True)
        model.fit(prophet_df)

        # Create a future DataFrame for 1 day into the future
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)

        # Extract the last forecasted value ('yhat')
        predicted_price = forecast.iloc[-1]['yhat']
        print(f"   [SUCCESS] Prophet forecast complete. Predicted Price: {predicted_price:.2f}")
        return float(predicted_price)

    except Exception as e:
        print(f"   [ERROR] An error occurred during Prophet forecasting: {e}")
        return np.nan

def lstm_forecast(df: pd.DataFrame, look_back_period: int = 60) -> float:
    """
    Generates a next-day price forecast using a simple LSTM neural network.

    Args:
        df (pd.DataFrame): DataFrame containing historical 'Close' prices.
        look_back_period (int): The number of previous time steps to use as input variables
                                to predict the next time period.

    Returns:
        float: The forecasted price for the next day. Returns np.nan on error.
    """
    if df.empty or len(df) <= look_back_period:
        print(f"   [WARN] LSTM forecast requires at least {look_back_period + 1} data points.")
        return np.nan

    print("   [INFO] Starting LSTM forecast...")
    try:
        # 1. Prepare the data
        data = df[['Close']].copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # 2. Create training dataset
        X_train, y_train = [], []
        for i in range(look_back_period, len(scaled_data)):
            X_train.append(scaled_data[i-look_back_period:i, 0])
            y_train.append(scaled_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # 3. Build and train the LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=16, epochs=5, verbose=0)

        # 4. Make a prediction
        last_sequence = scaled_data[-look_back_period:]
        last_sequence = np.reshape(last_sequence, (1, look_back_period, 1))
        
        predicted_price_scaled = model.predict(last_sequence, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)

        print(f"   [SUCCESS] LSTM forecast complete. Predicted Price: {predicted_price[0][0]:.2f}")
        return float(predicted_price[0][0])

    except Exception as e:
        print(f"   [ERROR] An error occurred during LSTM forecasting: {e}")
        return np.nan

if __name__ == '__main__':
    # This block allows for direct testing of this module
    print("--- Testing forecasting.py ---")
    test_coin = "ETH-USD"
    
    # We need data to test the forecast functions
    data = fetch_data(test_coin)

    if not data.empty:
        print(f"\n--- Testing with {test_coin} data ---")
        prophet_prediction = prophet_forecast(data)
        lstm_prediction = lstm_forecast(data)

        print("\n--- Test Results ---")
        print(f"Actual Last Price: {data['Close'].iloc[-1]:.2f}")
        print(f"Prophet Prediction: {prophet_prediction:.2f}" if prophet_prediction is not np.nan else "Prophet failed.")
        print(f"LSTM Prediction: {lstm_prediction:.2f}" if lstm_prediction is not np.nan else "LSTM failed.")
