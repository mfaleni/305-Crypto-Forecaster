import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect, Table, Column, MetaData, Integer, String, Float, DateTime

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("DATABASE_URL environment variable is not set.")

engine = create_engine(DATABASE_URL)
metadata = MetaData()

forecasts_table = Table('forecasts', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('Date', DateTime, nullable=False),
    Column('Coin', String, nullable=False),
    Column('Actual_Price', Float),
    Column('Prophet_Forecast', Float),
    Column('LSTM_Forecast', Float),
    Column('Sentiment_Score', Float),
    Column('RSI', Float),
    Column('MACD', Float),
    Column('All_Time_High', Float),
    Column('High_Forecast_5_Day', String),
    Column('analysis_summary', String),
    Column('analysis_hypothesis', String),
    Column('analysis_news_links', String)
)

def init_db():
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
    print("   [INFO] Saving forecast results to the database...")
    try:
        results_df['Date'] = pd.to_datetime(results_df['Date'])
        results_df.to_sql('forecasts', engine, if_exists='append', index=False)
        print(f"   [SUCCESS] Saved {len(results_df)} new records to the database.")
    except Exception as e:
        print(f"❌ [ERROR] Could not save results to database: {e}")
        raise

def load_forecast_results() -> pd.DataFrame:
    print("   [INFO] Loading forecast results from the database...")
    try:
        query = text("SELECT * FROM forecasts ORDER BY \"Date\" DESC")
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection)
        print(f"   [SUCCESS] Loaded {len(df)} records from the database.")
        return df
    except Exception as e:
        print(f"❌ [ERROR] Could not load results from database: {e}")
        return pd.DataFrame()