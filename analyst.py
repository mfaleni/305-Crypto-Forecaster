import os
import json
import openai

def get_daily_analysis(daily_briefing_data: dict) -> dict:
    """
    Takes a dictionary of the day's data, sends it to the GPT-4 API for analysis,
    and returns a structured JSON response.
    """
    print("   [INFO] Contacting AI Analyst (GPT-4)...")
    
    # 1. Prepare the data for the prompt
    coin = daily_briefing_data.get("coin_name")
    actual = daily_briefing_data.get("actual_price", 0)
    prophet = daily_briefing_data.get("prophet_forecast", 0)
    sentiment = daily_briefing_data.get("sentiment_score", 0)
    rsi = daily_briefing_data.get("rsi", 0)
    macd = daily_briefing_data.get("macd", 0)
    headlines = daily_briefing_data.get("top_headlines", [])

    delta = actual - prophet
    delta_percent = (delta / actual) * 100 if actual != 0 else 0
    
    headline_text = "\n".join([f"- {h['title']}" for h in headlines]) if headlines else "No headlines available."

    # 2. Craft the detailed prompt
    prompt = f"""
    Here is the daily market data for {coin}:
    - Actual Closing Price: ${actual:,.2f}
    - AI Forecasted Price: ${prophet:,.2f}
    - Delta (Actual - Forecast): ${delta:,.2f} ({delta_percent:.2f}%)
    - Market News Sentiment Score: {sentiment:.2f} (from -1.0 to 1.0)
    - RSI (Relative Strength Index): {rsi:.2f}
    - MACD (Moving Average Convergence Divergence): {macd:.2f}
    - Top News Headlines:
    {headline_text}

    Please act as an expert financial analyst. Your task is to provide a concise, data-driven analysis explaining the market dynamics for today.

    Based *only* on the data provided, generate a JSON object with the following three keys:
    1. "summary": A one-sentence summary of the day's forecast accuracy.
    2. "hypothesis": A 2-3 sentence hypothesis explaining the most likely reason for the delta between the actual price and the forecast. You MUST cite specific data points (e.g., "negative sentiment", "overbought RSI > 70", etc.) to support your hypothesis.
    3. "influential_headlines": From the list provided, choose up to 3 headlines you believe were most influential on the sentiment and price action. This must be an array of JSON objects, where each object has a "title" and "url" key.
    """

    try:
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst that provides structured data in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        response_content = completion.choices[0].message.content
        analysis = json.loads(response_content)
        
        # Ensure the news links are stored as a JSON string for the database
        analysis['news_links'] = json.dumps(analysis.get("influential_headlines", []))
        
        print("   [SUCCESS] AI analysis generated.")
        return analysis

    except Exception as e:
        print(f"❌ [ERROR] AI Analyst API call failed: {e}")
        return {
            "summary": "AI analysis could not be generated due to an API error.",
            "hypothesis": str(e),
            "news_links": json.dumps([])
        }