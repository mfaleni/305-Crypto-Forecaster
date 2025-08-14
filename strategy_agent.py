import os
import json
import openai
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_trade_recommendation(daily_briefing_data: dict) -> dict:
    """
    Sends the daily data briefing to the GPT-4 API, prompting it to act
    as an expert quantitative trader and return a structured trade recommendation.
    """
    logger.info("   [INFO] Contacting AI Strategy Agent (GPT-4)...")

    # Extract key data points for the prompt, ensuring robustness
    coin = daily_briefing_data.get("coin_name", "Unknown Coin")
    price = daily_briefing_data.get("actual_price", 0)
    
    # Forecasts
    prophet_forecast = daily_briefing_data.get("prophet_forecast", 0)
    lstm_forecast = daily_briefing_data.get("lstm_forecast", 0)

    # Technicals
    rsi = daily_briefing_data.get("rsi", 0)
    macd = daily_briefing_data.get("macd", 0)
    
    # Futures/Derivatives
    funding_rate = daily_briefing_data.get("funding_rate", 0)
    open_interest = daily_briefing_data.get("open_interest", 0)
    long_short_ratio = daily_briefing_data.get("long_short_ratio", 0)

    # On-Chain/Social
    mvrv = daily_briefing_data.get("mvrv_ratio", 0)
    sentiment = daily_briefing_data.get("sentiment_score", 0)
    galaxy_score = daily_briefing_data.get("galaxy_score", 0)

    # Construct the detailed prompt
    # This prompt forces the AI to consider the confluence of different data types.
    prompt = f"""
    You are an expert cryptocurrency quantitative analyst and trading strategist. Your task is to analyze the provided multi-source data for {coin} and generate a high-probability trade setup for the next 24-72 hour horizon.

    CRITICAL INSTRUCTION: Base your analysis *only* on the data provided.

    --- MARKET DATA SNAPSHOT ---

    [PRICE & FORECASTS]
    - Current Price: ${price:,.2f}
    - Prophet Forecast (24h): ${prophet_forecast:,.2f}
    - LSTM Forecast (24h): ${lstm_forecast:,.2f}

    [TECHNICAL INDICATORS]
    - RSI (14-day): {rsi:.2f} (Interpretation: <30 Oversold, >70 Overbought)
    - MACD (12, 26) Line: {macd:.4f}

    [DERIVATIVES DATA (CoinGlass)]
    - Funding Rate: {funding_rate:.4f}% (Interpretation: High positive suggests many longs/potential overheating; Negative suggests many shorts)
    - Open Interest: ${open_interest:,.0f}
    - Long/Short Ratio: {long_short_ratio:.2f}

    [ON-CHAIN & SOCIAL DATA]
    - MVRV Ratio (Santiment): {mvrv:.2f} (Interpretation: <1 Undervalued, >3.5 Overvalued)
    - News Sentiment Score: {sentiment:.2f} (-1.0 Bearish to 1.0 Bullish)
    - Galaxy Score (LunarCrush): {galaxy_score:.1f}/100

    --- ANALYSIS TASK ---
    Analyze the confluence between technical momentum, derivatives positioning, on-chain value, and social sentiment. Identify potential setups (e.g., trend continuation, mean reversion, squeeze potential).

    --- OUTPUT FORMAT ---
    Provide the recommendation as a structured JSON object with the following keys:

    1. "action": (String) "BUY", "SELL", or "HOLD".
    2. "entry_range": (String) The recommended price range for entry (e.g., "65000.00 - 65500.00"). If HOLD, use "N/A".
    3. "tp1": (Float) Take Profit Target 1 (Realistic short-term target).
    4. "tp2": (Float) Take Profit Target 2 (Optimistic target).
    5. "sl": (Float) Stop Loss (Critical risk management level; must be defined for BUY/SELL).
    6. "confidence": (Float) Confidence score from 0.0 (low confluence) to 1.0 (high confluence).
    7. "rationale": (String) A concise, 2-3 sentence explanation citing the specific data points supporting the recommendation.
    """

    # Define a safe default return structure
    default_response = {
        "action": "HOLD", "entry_range": "N/A", "tp1": 0.0, "tp2": 0.0, "sl": 0.0, "confidence": 0.0,
        "rationale": "Strategy generation failed."
    }

    try:
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4-turbo", # Use a capable model for complex analysis
            messages=[
                {"role": "system", "content": "You are a quantitative trading strategist that provides structured trade setups exclusively in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3 # Lower temperature for more deterministic, analytical outputs
        )

        response_content = completion.choices[0].message.content
        recommendation = json.loads(response_content)

        # Validate the output structure
        required_keys = ["action", "entry_range", "tp1", "tp2", "sl", "confidence", "rationale"]
        if not all(key in recommendation for key in required_keys):
            logger.error(f"   [ERROR] AI response missing required keys. Response: {response_content}")
            default_response["rationale"] = "Strategy generation failed: Invalid JSON structure."
            return default_response

        logger.info("   [SUCCESS] AI trade recommendation generated.")
        return recommendation

    except Exception as e:
        logger.error(f" ❌  [ERROR] AI Strategy Agent process failed: {e}")
        default_response["rationale"] = f"Strategy generation failed due to API or processing error: {e}"
        return default_response