import logging
import requests
import pandas as pd
from datetime import timedelta
from backend.tools.utils import parse_structured_input, GNEWS_API_KEY

logger = logging.getLogger(__name__)

def get_gnews_articles(user_query: str):

    try:
        data = parse_structured_input(user_query)

        logger.info(f"Parsed structured input for news tool: {data}")

        company = data["company"]
        date_str = data["date"]

        url = "https://gnews.io/api/v4/search"

        params = {
            "q": company,
            "lang": "en",
            "country": "in",
            "max": 20,
            "apikey": GNEWS_API_KEY,
            "from": (pd.Timestamp(date_str) - timedelta(days=7)).strftime("%Y-%m-%d"),
            "to": date_str
        }

        logger.info(f"Making API call to GNews with params: {params}")

        response = requests.get(url, params=params)
        response.raise_for_status()

        news_data = response.json()

        articles = []

        for article in news_data.get("articles", []):
            articles.append({
                "title": article["title"],
                "description": article["description"],
                "url": article["url"],
                "published_at": article["publishedAt"],
                "source": article["source"]["name"]
            })


        logger.info(f"Retrieved {len(articles)} articles for {company} around {date_str}")  
        
        return articles

    except Exception as e:
        logger.error(e)
        return []