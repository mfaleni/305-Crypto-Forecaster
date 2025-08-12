# analyst.py

import os
import json
import openai

# --- Ensure OpenAI client is configured ---
# This setup assumes the API key is loaded from the .env file in daily_runner.py
try:
    client = openai.OpenAI()
except openai.OpenAIError:
    print("❌ [FATAL] OpenAI API key not configured. Please check your .env file.")
    client = None

def get_daily_analysis(daily_briefing_data: dict) -> dict:
    """
    Takes a comprehensive dictionary of the day's market data, sends it to the
    GPT-4 model for in-depth analysis, and returns a structured market report.
    """
    if not client:
        return {
            "summary": "AI analysis failed because the OpenAI client is not configured.",
            "hypothesis": "Configuration error.",
            "news_links": "[]"
        }

    coin_name = daily_briefing_data.get("coin_name", "the asset")
    print(f"   [INFO] Briefing AI Analyst for comprehensive report on {coin_name}...")

    # --- 1. Construct the UPGRADED Prompt ---
    # This new prompt instructs the AI to generate a detailed, multi-part report.
    system_prompt = """
    You are an expert crypto market analyst writing a daily briefing. Your tone is objective, data-driven, and insightful. Your task is to synthesize a comprehensive set of market data into a multi-part report.

    You MUST provide your response in a single, valid JSON object with the following five keys:
    1. "title": A compelling, news-style headline for today's analysis (e.g., "Ethereum Navigates On-Chain Strength Amidst Overheated Futures Market").
    2. "price_action_recap": A 1-2 sentence summary of the recent price action, mentioning key support or resistance levels being tested.
    3. "bullish_case": A markdown-formatted string. Detail the bullish signals from the provided data. For each point, start with a bolded title (e.g., "**On-Chain Accumulation**"), cite the specific metric (e.g., MVRV Ratio, Exchange Netflow), and explain its positive implication.
    4. "bearish_case": A markdown-formatted string. Detail the bearish signals. Follow the same format as the bullish case, titling each point (e.g., "**Overheated Derivatives**") and citing the relevant data (e.g., Funding Rate).
    5. "analyst_hypothesis": A concluding 2-3 sentence paragraph. Synthesize the conflicting bullish and bearish cases to form a primary, forward-looking hypothesis about the market's likely short-term direction.
    """

    # We provide the full briefing data to the AI.
    user_prompt = f"""
    Generate a comprehensive market analysis report for {coin_name} based on the following data.
    Directly cite the data points in your analysis.

    ```json
    {json.dumps(daily_briefing_data, indent=2)}
    ```
    """

    try:
        # --- 2. Make the API Call to GPT-4 ---
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4 # A balance between creativity and factual analysis
        )

        # --- 3. Parse and Prepare the Response ---
        analysis_from_ai = json.loads(completion.choices[0].message.content)

        # For the database, we'll combine the structured cases into one summary field.
        # This keeps the database schema simpler while providing rich text for the frontend.
        summary_for_db = f"### Bullish Case\n{analysis_from_ai.get('bullish_case', '')}\n\n### Bearish Case\n{analysis_from_ai.get('bearish_case', '')}"

        # We will also need to update the database schema and daily_runner to save these new fields.
        # For now, we'll structure the output dictionary.
        analysis_to_save = {
            # Old fields for compatibility (can be removed later)
            "summary": summary_for_db,
            "hypothesis": analysis_from_ai.get("analyst_hypothesis", "No hypothesis generated."),
            "news_links": json.dumps(daily_briefing_data.get("top_headlines", [])),

            # New structured fields for the upgraded dashboard
            "report_title": analysis_from_ai.get("title", "Daily Analysis"),
            "report_recap": analysis_from_ai.get("price_action_recap", ""),
            "report_bullish": analysis_from_ai.get("bullish_case", ""),
            "report_bearish": analysis_from_ai.get("bearish_case", ""),
            "report_hypothesis": analysis_from_ai.get("analyst_hypothesis", "")
        }

        print(f"   [SUCCESS] Comprehensive AI analysis for {coin_name} generated.")
        return analysis_to_save

    except Exception as e:
        print(f"❌ [ERROR] AI Analyst API call failed: {e}")
        return {
            "summary": "AI analysis could not be generated due to an API error.",
            "hypothesis": str(e),
            "news_links": "[]",
            "report_title": "Analysis Failed",
            "report_recap": "", "report_bullish": "", "report_bearish": "", "report_hypothesis": ""
        }