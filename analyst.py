import os
import json
import openai

try:
    client = openai.OpenAI()
except openai.OpenAIError:
    print("❌ [FATAL] OpenAI API key not configured. Please check your .env file.")
    client = None

def get_daily_analysis(daily_briefing_data: dict) -> dict:
    if not client:
        return { "summary": "AI analysis failed.", "hypothesis": "Configuration error.", "news_links": "[]" }
        
    coin_name = daily_briefing_data.get("coin_name", "the asset")
    print(f"   [INFO] Briefing AI Analyst for comprehensive report on {coin_name}...")

    # --- START: UPGRADED SYSTEM PROMPT ---
    system_prompt = """
    You are an expert crypto market analyst writing a daily briefing. Your tone is objective, data-driven, and insightful. Your task is to synthesize a comprehensive set of market data into a multi-part report.

    You MUST provide your response in a single, valid JSON object with the following five keys:
    1. "title": A compelling, news-style headline for today's analysis.
    2. "price_action_recap": A 1-2 sentence summary of the recent price action.
    3. "bullish_case": A markdown-formatted string. Detail the bullish signals. For each point, start with a bolded title (e.g., "**On-Chain Strength**"), cite specific metrics (e.g., MVRV Ratio, Daily Active Addresses, positive Exchange Supply Ratio), and explain the positive implication.
    4. "bearish_case": A markdown-formatted string. Detail the bearish signals. Follow the same format, using bolded titles (e.g., "**Overheated Derivatives Market**") and citing specific metrics (e.g., High Leverage Ratio, Funding Rates).
    5. "analyst_hypothesis": A concluding 2-3 sentence paragraph. Synthesize the conflicting bullish and bearish cases, giving special weight to derivatives and on-chain data, to form a primary, forward-looking hypothesis.
    """
    # --- END: UPGRADED SYSTEM PROMPT ---

    user_prompt = f"""
    Generate a comprehensive market analysis report for {coin_name} based on the following data.
    Directly cite the data points in your analysis, especially the advanced metrics.

    ```json
    {json.dumps(daily_briefing_data, indent=2)}
    ```
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4
        )
        
        analysis_from_ai = json.loads(completion.choices[0].message.content)
        
        summary_for_db = f"### Bullish Case\n{analysis_from_ai.get('bullish_case', '')}\n\n### Bearish Case\n{analysis_from_ai.get('bearish_case', '')}"

        analysis_to_save = {
            "summary": summary_for_db,
            "hypothesis": analysis_from_ai.get("analyst_hypothesis", "No hypothesis generated."),
            "news_links": json.dumps(daily_briefing_data.get("top_headlines", [])),
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
            "summary": "AI analysis could not be generated.", "hypothesis": str(e), "news_links": "[]",
            "report_title": "Analysis Failed", "report_recap": "", "report_bullish": "", "report_bearish": "", "report_hypothesis": ""
        }