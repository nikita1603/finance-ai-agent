"""GNews integration helper.

Provides `get_gnews_articles(user_query: str)` which queries the GNews
API for articles related to a company around a specified date. The
function returns a list of lightweight article dicts or an empty list on
failure. No runtime logic was changed; only documentation and comments
were added for clarity.
"""

import logging
import requests
import pandas as pd
from datetime import timedelta
from backend.tools.utils import parse_structured_input, GNEWS_API_KEY

logger = logging.getLogger(__name__)


def get_gnews_articles(user_query: str):
    """Fetch recent news articles for the company in `user_query`.

    Expected structured input fields (parsed by `parse_structured_input`):
        - Company: company name (used as query term)
        - Date: YYYY-MM-DD (end of the search window)

    The function searches a 7-day window ending on `Date` and returns a
    list of article dicts with `title`, `description`, `url`,
    `published_at`, and `source` keys.
    """

    try:
        # Extract structured fields from incoming query
        data = parse_structured_input(user_query)

        logger.info(f"Parsed structured input for news tool: {data}")

        company = data["company"]
        date_str = data["date"]

        url = "https://gnews.io/api/v4/search"

        # Build request parameters: search company in English within India,
        # limit to 20 results, and search a 7-day window ending on date_str
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

        # Normalize returned articles into a compact dict list
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
        # Log the exception and return an empty list to indicate no data
        logger.error(e)
        return []