import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect, Table, Column, MetaData, Integer, String, Float, DateTime

# --- START: MODIFIED DATABASE CONNECTION LOGIC ---
# This new logic intelligently connects to Render (cloud) or Docker (local)

DATABASE_URL = os.getenv("DATABASE_URL")
engine = None

if DATABASE_URL:
    # If a DATABASE_URL is provided (for Render), use it directly.
    print("   [INFO] Connecting to cloud database (Render)...")
    engine = create_engine(DATABASE_URL)
else:
    # If DATABASE_URL is not set, build the URL from local DB credentials.
    print("   [INFO] Connecting to local database (Docker)...")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    
    # Check if all required local DB variables are present
    if not all([db_user, db_pass, db_host, db_port, db_name]):
        raise Exception("Missing one or more required local database environment variables (DB_USER, DB_PASSWORD, etc.).")
    
    local_db_url = f"postgresql+psycopg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(local_db_url)

# --- END: MODIFIED DATABASE CONNECTION LOGIC ---


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
    Column('Funding_Rate', Float),
    Column('Open_Interest', Float),
    Column('Long_Short_Ratio', Float),
    Column('MVRV_Ratio', Float),
    Column('Social_Dominance', Float),
    Column('Daily_Active_Addresses', Float),
    Column('Galaxy_Score', Float),
    Column('Alt_Rank', Float),
    Column('analysis_summary', String),
    Column('analysis_hypothesis', String),
    Column('analysis_news_links', String),
    Column('report_title', String),
    Column('report_recap', String),
    Column('report_bullish', String),
    Column('report_bearish', String),
    Column('report_hypothesis', String),
    Column('user_feedback', String),
    Column('user_correction', String)
)

def init_db():
    print("   [INFO] Initializing database...")
    try:
        metadata.create_all(engine, checkfirst=True)
        print("   [SUCCESS] Database initialized and schema verified.")
    except Exception as e:
        print(f"❌ [ERROR] Could not initialize database: {e}")
        raise

def save_forecast_results(results_df: pd.DataFrame):
    print("   [INFO] Saving forecast results to the database...")
    try:
        results_df['Date'] = pd.to_datetime(results_df['Date'])
        # Add a check for the 'Exchange_Net_Flow' column before saving
        if 'Exchange_Net_Flow' not in forecasts_table.columns:
             results_df = results_df.drop(columns=['Exchange_Net_Flow'], errors='ignore')
        results_df.to_sql('forecasts', engine, if_exists='append', index=False)
        print(f"   [SUCCESS] Saved {len(results_df)} new records to the database.")
    except Exception as e:
        print(f"❌ [ERROR] Could not save results to database: {e}")
        raise

def load_forecast_results() -> pd.DataFrame:
    print("   [INFO] Loading forecast results from the database...")
    try:
        query = text("SELECT * FROM forecasts ORDER BY \"Date\" DESC, id DESC")
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection)
        print(f"   [SUCCESS] Loaded {len(df)} records from the database.")
        return df
    except Exception as e:
        print(f"❌ [ERROR] Could not load results from database: {e}")
        return pd.DataFrame()

def update_feedback(record_id: int, feedback: str, correction: str = ""):
    print(f"   [INFO] Updating feedback for record ID: {record_id}...")
    try:
        with engine.connect() as connection:
            stmt = text(
                "UPDATE forecasts SET user_feedback = :feedback, user_correction = :correction WHERE id = :id"
            )
            connection.execute(stmt, {"feedback": feedback, "correction": correction, "id": record_id})
            connection.commit()
        print("   [SUCCESS] Feedback updated in the database.")
        return True
    except Exception as e:
        print(f"❌ [ERROR] Could not update feedback in database: {e}")
        return False