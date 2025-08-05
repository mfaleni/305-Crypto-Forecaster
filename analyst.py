import os
import json
import openai

def get_daily_analysis(daily_briefing_data: dict) -> dict:
    """
    Takes a dictionary of the day's data, sends it to the GPT-4 API for analysis,
    and returns a structured JSON response with validated URLs.
    """
    print("   [INFO] Contacting AI Analyst (GPT-4)...")
    
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

    prompt = f"""
    Here is the daily market data for {coin}:
    - Actual Closing Price: ${actual:,.2f}
    - AI Forecasted Price: ${prophet:,.2f}
    - Delta (Actual - Forecast): ${delta:,.2f} ({delta_percent:.2f}%)
    - Market News Sentiment Score: {sentiment:.2f}
    - RSI: {rsi:.2f}
    - MACD: {macd:.2f}
    - Top News Headlines Provided:
    {headline_text}

    Please act as an expert financial analyst. Your task is to provide a concise, data-driven analysis.

    Based *only* on the data provided, generate a JSON object with the following three keys:
    1. "summary": A one-sentence summary of the day's forecast accuracy.
    2. "hypothesis": A 2-3 sentence hypothesis explaining the most likely reason for the delta, citing specific data points to support your hypothesis.
    3. "influential_headline_titles": An array of up to 3 strings, where each string is the exact title of a headline from the list provided that you believe was most influential.
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
        analysis_from_ai = json.loads(response_content)
        
        # --- ROBUST URL MAPPING ---
        # Look up the original URLs based on the titles returned by the AI
        final_headlines = []
        ai_titles = analysis_from_ai.get("influential_headline_titles", [])
        for title in ai_titles:
            for original_headline in headlines:
                if original_headline["title"] == title:
                    final_headlines.append(original_headline)
                    break # Move to the next AI title
        
        # Prepare the final analysis object for the database
        analysis_to_save = {
            "summary": analysis_from_ai.get("summary"),
            "hypothesis": analysis_from_ai.get("hypothesis"),
            "news_links": json.dumps(final_headlines)
        }
        
        print("   [SUCCESS] AI analysis generated and URLs verified.")
        return analysis_to_save

    except Exception as e:
        print(f"❌ [ERROR] AI Analyst API call failed: {e}")
        return {
            "summary": "AI analysis could not be generated due to an API error.",
            "hypothesis": str(e),
            "news_links": json.dumps([])
        }