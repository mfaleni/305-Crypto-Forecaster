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
        return {}
        
    coin_name = daily_briefing_data.get("coin_name", "the asset")
    print(f"   [INFO] Briefing AI Analyst for comprehensive report on {coin_name}...")

    # This is the new, more powerful prompt
    system_prompt = """
    You are a world-class quantitative analyst and a highly successful crypto trader, operating like a supercomputer. Your task is to synthesize a comprehensive set of market data into a detailed, multi-part trading report. You must be objective, data-driven, and provide clear, actionable insights.

    You MUST provide your response in a single, valid JSON object with the following keys:
    1. "report_title": A compelling, news-style headline for today's analysis.
    2. "price_action_recap": A 1-2 sentence summary of the recent price action, referencing key price levels.
    3. "bullish_case": A markdown-formatted string. Detail the bullish signals from the provided data. For each point, start with a bolded title (e.g., "**On-Chain Strength**"), cite specific metrics (e.g., MVRV Ratio, Daily Active Addresses), and explain the positive implication.
    4. "bearish_case": A markdown-formatted string. Detail the bearish signals. Follow the same format, using bolded titles (e.g., "**Overheated Derivatives Market**") and citing specific metrics (e.g., High Leverage Ratio, Funding Rates).
    5. "analyst_hypothesis": A concluding 2-3 sentence paragraph. Synthesize the conflicting bullish and bearish cases to form a primary, forward-looking hypothesis.
    6. "strategic_outlook": A markdown-formatted string with two sub-sections: '**Potential Entry Point**' and '**Potential Exit Point**'. For each, provide a specific price target and a detailed 5-7 sentence justification, citing multiple data points from the briefing such as Fibonacci levels, RSI extremes, historical support/resistance, and market sentiment to explain your reasoning.
    7. "influential_headline_titles": An array of the exact titles of the top 3 most influential headlines from the list provided.
    """

    user_prompt = f"""
    Generate a comprehensive market analysis and trading plan for {coin_name} based on the following data.
    Your analysis must be thorough and your strategic outlook must be detailed and well-justified.

    ```json
    {json.dumps(daily_briefing_data, indent=2, default=str)}
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
            temperature=0.5
        )
        
        analysis_from_ai = json.loads(completion.choices[0].message.content)
        
        # Map influential titles back to full headline objects for the database
        final_headlines = []
        ai_titles = analysis_from_ai.get("influential_headline_titles", [])
        for title in ai_titles:
            for original_headline in daily_briefing_data.get("top_headlines", []):
                if original_headline["title"] == title:
                    final_headlines.append(original_headline)
                    break
        
        # Combine bullish and bearish cases for the old summary field for compatibility
        summary_for_db = f"### Bullish Case\n{analysis_from_ai.get('bullish_case', '')}\n\n### Bearish Case\n{analysis_from_ai.get('bearish_case', '')}"

        analysis_to_save = {
            "analysis_summary": summary_for_db,
            "analysis_hypothesis": analysis_from_ai.get("analyst_hypothesis"),
            "analysis_news_links": json.dumps(final_headlines),
            "report_title": analysis_from_ai.get("report_title"),
            "report_recap": analysis_from_ai.get("price_action_recap"),
            "report_bullish": analysis_from_ai.get("bullish_case"),
            "report_bearish": analysis_from_ai.get("bearish_case"),
            "report_hypothesis": analysis_from_ai.get("analyst_hypothesis"),
            "strategic_outlook": analysis_from_ai.get("strategic_outlook")
        }

        print(f"   [SUCCESS] Comprehensive AI analysis for {coin_name} generated.")
        return analysis_to_save

    except Exception as e:
        print(f"❌ [ERROR] AI Analyst API call failed: {e}")
        return {}