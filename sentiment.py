import os
import requests
import openai
import re
from datetime import datetime, timedelta

def get_news_sentiment(coin_ticker: str, coin_name: str, api_key: str) -> float:
    """
    Fetches recent news for a cryptocurrency and returns a sentiment score from GPT.
    Relies on the calling script (daily_runner) to have set the OpenAI API key.
    """
    print(f"   [INFO] Starting sentiment analysis for {coin_ticker}...")
    try:
        # --- Step 1: Fetch news articles using the provided key ---
        if not api_key:
            print("   [WARN] NewsAPI key was not provided. Skipping sentiment analysis.")
            return 0.0

        from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        url = (f'https://newsapi.org/v2/everything?'
               f'q={coin_name}&'
               f'from={from_date}&'
               f'sortBy=publishedAt&'
               f'language=en&'
               f'apiKey={api_key}')

        response = requests.get(url)
        response.raise_for_status() 
        
        articles = response.json().get("articles", [])
        if not articles:
            print(f"   [WARN] No recent news articles found for {coin_name}.")
            return 0.0

        headlines = [f"Title: {a['title']}. Desc: {a.get('description', '')}" for a in articles[:10]]
        news_text = "\n".join(headlines)

        # --- Step 2: Score sentiment with OpenAI ---
        # The openai.api_key is expected to be set globally by the runner
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial sentiment analyst. Based on the following news headlines, provide a single sentiment score from -1.0 (very negative) to 1.0 (very positive) for the cryptocurrency mentioned. Respond with only the numerical score and nothing else."},
                {"role": "user", "content": f"Analyze the sentiment for {coin_name} from these articles:\n\n{news_text}"}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        content = completion.choices[0].message.content
        
        match = re.search(r"(-?\d+\.?\d*)", content)
        if match:
            score = float(match.group(0))
            final_score = max(-1.0, min(1.0, score))
            print(f"   [SUCCESS] Sentiment analysis complete. Score: {final_score:.2f}")
            return final_score
        else:
            print(f"   [WARN] Could not parse sentiment score from OpenAI response: '{content}'")
            return 0.0

    except requests.exceptions.RequestException as e:
        print(f"   [ERROR] NewsAPI request failed: {e}")
        return 0.0
    except Exception as e:
        print(f"   [ERROR] An error occurred during sentiment analysis: {e}")
        return 0.0