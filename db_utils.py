import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect, Table, Column, MetaData, Integer, String, Float, DateTime

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("DATABASE_URL environment variable is not set.")

engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Define the structure of our forecasts table
forecasts_table = Table('forecasts', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('Date', DateTime, nullable=False),
    Column('Coin', String, nullable=False),
    Column('Actual_Price', Float),
    Column('Prophet_Forecast', Float),
    Column('LSTM_Forecast', Float),
    Column('Sentiment_Score', Float),
    Column('RSI', Float), # <--- CHANGE: ADDED THIS LINE --->
    Column('MACD', Float), # <--- CHANGE: ADDED THIS LINE --->
    Column('All_Time_High', Float),
    Column('High_Forecast_5_Day', String) # Storing as a JSON string
)

def init_db():
    """
    Initializes the database and creates the 'forecasts' table if it doesn't exist.
    """
    print("   [INFO] Initializing database...")
    try:
        inspector = inspect(engine)
        if not inspector.has_table('forecasts'):
            print("   [INFO] 'forecasts' table not found. Creating table...")
            metadata.create_all(engine)
            print("   [SUCCESS] 'forecasts' table created.")
        else:
            print("   [INFO] 'forecasts' table already exists.")
    except Exception as e:
        print(f"❌ [ERROR] Could not initialize database: {e}")
        raise

def save_forecast_results(results_df: pd.DataFrame):
    """
    Saves a DataFrame of forecast results to the database.
    """
    print("   [INFO] Saving forecast results to the database...")
    try:
        # Ensure the 'Date' column is in the correct format
        results_df['Date'] = pd.to_datetime(results_df['Date'])
        results_df.to_sql('forecasts', engine, if_exists='append', index=False)
        print(f"   [SUCCESS] Saved {len(results_df)} new records to the database.")
    except Exception as e:
        print(f"❌ [ERROR] Could not save results to database: {e}")
        raise

def load_forecast_results() -> pd.DataFrame:
    """
    Loads