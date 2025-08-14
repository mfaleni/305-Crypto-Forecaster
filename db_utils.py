import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect, Table, Column, MetaData, Integer, String, Float, DateTime
from dotenv import load_dotenv
import logging

# Load environment variables (important for local runs)
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Intelligent Database Connection Logic (Retained from your codebase) ---
DATABASE_URL = os.getenv("DATABASE_URL")
engine = None

if DATABASE_URL:
    logger.info("   [INFO] Connecting to cloud database (Render/External)...")
    # Ensure the URL uses the modern psycopg driver for SQLAlchemy
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
    engine = create_engine(DATABASE_URL)
else:
    logger.info("   [INFO] Connecting to local database (Docker/Localhost)...")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    
    if not all([db_user, db_pass, db_host, db_port, db_name]):
        raise Exception("Missing local database environment variables (DB_USER, etc.) and DATABASE_URL is not set.")
    
    local_db_url = f"postgresql+psycopg://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(local_db_url)

if engine is None:
    raise Exception("Database engine could not be initialized.")

metadata = MetaData()

# --- Unified Table Schema Definition ---
forecasts_table = Table('forecasts', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('Date', DateTime, nullable=False),
    Column('Coin', String, nullable=False),
    
    # --- Price and Forecast Data ---
    Column('Actual_Price', Float),
    Column('Prophet_Forecast', Float),
    Column('LSTM_Forecast', Float),
    Column('High_Forecast_5_Day', String),
    
    # --- Technical & Fundamental Indicators (Standard) ---
    Column('Sentiment_Score', Float),
    Column('RSI', Float),
    Column('MACD', Float),
    Column('All_Time_High', Float),
    
    # --- CoinGlass Futures Data ---
    Column('Funding_Rate', Float),
    Column('Open_Interest', Float),
    Column('Long_Short_Ratio', Float),
    
    # --- Santiment On-Chain/Social Data ---
    Column('MVRV_Ratio', Float),
    Column('Social_Dominance', Float),
    Column('Daily_Active_Addresses', Float),
    
    # --- LunarCrush Social Data ---
    Column('Galaxy_Score', Float),
    Column('Alt_Rank', Float),
    
    # --- Advanced Metrics (Existing additions you requested to keep) ---
    Column('Leverage_Ratio', Float),
    Column('Futures_Volume_24h', Float),
    Column('Exchange_Supply_Ratio', Float),
    Column('Exchange_Net_Flow', Float), # (From the guide, kept here)

    # --- AI Analysis & Feedback (From the Guide) ---
    Column('analysis_summary', String),
    Column('analysis_hypothesis', String),
    Column('analysis_news_links', String),
    Column('user_feedback', String),
    Column('user_correction', String),

    # --- AI Reports (Existing additions you requested to keep) ---
    Column('report_title', String),
    Column('report_recap', String),
    Column('report_bullish', String),
    Column('report_bearish', String),
    Column('report_hypothesis', String),

    # --- NEW: AI Trade Recommendations (The Strategy Agent) ---
    Column('trade_action', String),          # BUY, SELL, HOLD
    Column('trade_entry_range', String),     # e.g., "65000.00 - 65500.00"
    Column('trade_tp1', Float),              # Take Profit 1
    Column('trade_tp2', Float),              # Take Profit 2
    Column('trade_sl', Float),               # Stop Loss
    Column('trade_confidence', Float),       # 0.0 to 1.0
    Column('trade_rationale', String)        # AI's reasoning for the trade
)

# === Database Interaction Functions ===
# These functions utilize the dynamically created 'engine' variable.

def init_db():
    logger.info("   [INFO] Initializing database...")
    try:
        inspector = inspect(engine)
        if not inspector.has_table('forecasts'):
            logger.info("   [INFO] 'forecasts' table not found. Creating table...")
            metadata.create_all(engine)
            logger.info("   [SUCCESS] 'forecasts' table created.")
        else:
            logger.info("   [INFO] 'forecasts' table already exists.")
    except Exception as e:
        logger.error(f" ❌  [ERROR] Could not initialize database: {e}")
        raise

def save_forecast_results(results_df: pd.DataFrame):
    logger.info("   [INFO] Saving forecast results to the database...")
    try:
        # Ensure the Date column is properly typed before insertion
        results_df['Date'] = pd.to_datetime(results_df['Date'])
        results_df.to_sql('forecasts', engine, if_exists='append', index=False)
        logger.info(f"   [SUCCESS] Saved {len(results_df)} new records to the database.")
    except Exception as e:
        logger.error(f" ❌  [ERROR] Could not save results to database: {e}")
        raise

def load_forecast_results() -> pd.DataFrame:
    logger.info("   [INFO] Loading forecast results from the database...")
    try:
        # Use text() for SQL queries for better compatibility and explicit ordering
        query = text("SELECT * FROM forecasts ORDER BY \"Date\" DESC, id DESC")
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection)
        
        # Convert Date column back to datetime objects after loading
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
        logger.info(f"   [SUCCESS] Loaded {len(df)} records from the database.")
        return df
    except Exception as e:
        logger.error(f" ❌  [ERROR] Could not load results from database: {e}")
        return pd.DataFrame() # Return empty DataFrame on failure

def update_feedback(record_id: int, feedback: str, correction: str = ""):
    logger.info(f"   [INFO] Updating feedback for record ID: {record_id}...")
    try:
        # Use a transaction for safe updates
        with engine.connect() as connection:
            transaction = connection.begin()
            stmt = text(
                "UPDATE forecasts SET user_feedback = :feedback, user_correction = :correction WHERE id = :id"
            )
            connection.execute(stmt, {"feedback": feedback, "correction": correction, "id": record_id})
            transaction.commit() # Commit the transaction
        logger.info("   [SUCCESS] Feedback updated in the database.")
        return True
    except Exception as e:
        logger.error(f" ❌  [ERROR] Could not update feedback in database: {e}")
        return False