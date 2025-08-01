import os
import requests
import openai
import re
from datetime import datetime, timedelta

def get_news_sentiment(coin_ticker: str, coin_name: str) -> float:
    """
    Fetches recent news for a cryptocurrency and returns a sentiment score from GPT.

    This function orchestrates fetching news from NewsAPI and then scoring
    the sentiment of the article headlines using OpenAI's GPT model.

    Args:
        coin_ticker (str): The ticker symbol (e.g., "BTC-USD"). Used for logging.
        coin_name (str): The full name of the coin (e.g., "Bitcoin"). Used for the news query.

    Returns:
        float: A sentiment score between -1.0 (very negative) and 1.0 (very positive).
               Returns 0.0 if the process fails at any step.
    """
    print(f"   [INFO] Starting sentiment analysis for {coin_ticker}...")
    try:
        # --- Step 1: Fetch news articles ---
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            print("   [WARN] NEWS_API_KEY environment variable not set. Skipping sentiment analysis.")
            return 0.0

        # Search for news from the last 3 days to ensure we get results
        from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        # Construct the NewsAPI query
        url = (f'https://newsapi.org/v2/everything?'
               f'q={coin_name}&'
               f'from={from_date}&'
               f'sortBy=publishedAt&'
               f'language=en&'
               f'apiKey={api_key}')

        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        articles = response.json().get("articles", [])
        if not articles:
            print(f"   [WARN] No recent news articles found for {coin_name}.")
            return 0.0

        # Combine the headlines and descriptions into one text block for analysis
        headlines = [f"Title: {a['title']}. Desc: {a.get('description', '')}" for a in articles[:10]] # Limit to 10 articles
        news_text = "\n".join(headlines)

        # --- Step 2: Score sentiment with OpenAI ---
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            print("   [WARN] OPENAI_API_KEY environment variable not set. Skipping sentiment analysis.")
            return 0.0
            
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial sentiment analyst. Based on the following news headlines, provide a single sentiment score from -1.0 (very negative) to 1.0 (very positive) for the cryptocurrency mentioned. Respond with only the numerical score and nothing else."},
                {"role": "user", "content": f"Analyze the sentiment for {coin_name} from these articles:\n\n{news_text}"}
            ],
            temperature=0.0, # Low temperature for consistent, factual scoring
            max_tokens=10
        )
        
        content = completion.choices[0].message.content
        
        # Use regex to reliably extract the floating-point number
        match = re.search(r"(-?\d+\.?\d*)", content)
        if match:
            score = float(match.group(0))
            # Clamp the score to be strictly between -1.0 and 1.0
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

if __name__ == '__main__':
    # This block allows for direct testing of this module
    # Make sure to set your API keys as environment variables before running
    # export OPENAI_API_KEY="your_key"
    # export NEWS_API_KEY="your_key"
    print("--- Testing sentiment.py ---")
    
    # Test with a valid coin
    sentiment_score = get_news_sentiment(coin_ticker="BTC-USD", coin_name="Bitcoin")
    print("\n--- Test Result ---")
    print(f"Final sentiment score for Bitcoin: {sentiment_score}")

