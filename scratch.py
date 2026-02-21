from time import strptime
from ddgs import DDGS 
import yfinance as yf
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)

GNEWS_API_KEY="b56aee6c73868a137197b2e2db2b41a3"


# Indian news sites to prioritize for finance-related queries
INDIA_FINANCE_SITES = [
    "nseindia.com",
    "bseindia.com",
    "moneycontrol.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "financialexpress.com",
    "cnbctv18.com",
    "zeebiz.com",
    "groww.in",
    "tickertape.in"
]

def india_finance_search_tool(user_query: str) -> str:
    """
    Searches ONLY Indian finance and stock market websites.
    """
    lines = user_query.split("\n")
    date = lines[0].replace("Date:", "").strip()
    company = lines[1].replace("Company:", "").strip()
    question = lines[2].replace("Question:", "").strip()

    final_query = f"{date} {company} {question}"

    logger.info(f"Using Indian Finance Search Tool for query: {final_query}")


    results_text = []

    with DDGS() as ddgs:
        for site in INDIA_FINANCE_SITES:
            site_query = f"site:{site} {final_query}"
            results = ddgs.text(site_query, max_results=2)

            for r in results:
                snippet = f"""
Source: {site}
Title: {r['title']}
Snippet: {r['body']}
Link: {r['href']}
"""
                results_text.append(snippet)

    if not results_text:

        logger.info(f"No results found")

        return "No Indian finance-specific results found."
    
    logger.info(f"Indian Finance Search Tool found {len(results_text)} results")

    return "\n\n".join(results_text[:8])



#stock price retrieval tool
TICKER_MAP = {
    "Hdfc": ["HDFCBANK.NS"],
    "Reliance": ["RELIANCE.NS"],
}


def stock_price_tool(user_query: str):
    """
    Accepts full user query.
    Extracts company and date.
    Returns price data + key market fundamentals.
    """

    try:
        logger.info(f"Processing stock price query: {user_query}")

        lines = user_query.split("\n")
        date = lines[0].replace("Date:", "").strip()
        company = lines[1].replace("Company:", "").strip()

        tickers = TICKER_MAP.get(company)

        if not tickers:
            return f"No ticker mapping found for {company}."

        selected_date = datetime.strptime(date, "%Y-%m-%d")
        next_day = selected_date + timedelta(days=1)

        results = {}

        for ticker_symbol in tickers:

            stock = yf.Ticker(ticker_symbol)

            # -------- PRICE DATA --------
            data = stock.history(start=selected_date, end=next_day)

            price_data = {}

            if not data.empty:
                open_price = round(float(data["Open"].iloc[0]), 2)
                high_price = round(float(data["High"].iloc[0]), 2)
                low_price = round(float(data["Low"].iloc[0]), 2)
                close_price = round(float(data["Close"].iloc[0]), 2)
                volume = int(data["Volume"].iloc[0])

                pct_change = round(((close_price - open_price) / open_price) * 100, 2)
                intraday_range = round(high_price - low_price, 2)

                price_data = {
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume,
                    "Percentage Change (%)": pct_change,
                    "Intraday Range": intraday_range
                }

            else:
                logger.info(f"No trading data for {ticker_symbol} on {date}")

            # -------- FUNDAMENTALS --------
            try:
                info = stock.info
                shares_outstanding = info.get("sharesOutstanding")
                if shares_outstanding and price_data.get("Close"):
                    market_cap = round(price_data["Close"] * shares_outstanding, 2)

                fundamentals = {
                    "Market Cap": market_cap,
                    "Shares Outstanding": shares_outstanding,
                    "Trailing PE": info.get("trailingPE"),
                    "Price to Book": info.get("priceToBook"),
                    "Dividend Yield": info.get("dividendYield"),
                }
            except Exception:
                logger.warning(f"Fundamental fetch failed: {e}")
                fundamentals = {"Error": "Fundamental data unavailable"}

            results[ticker_symbol] = {
                "Date": date,
                "Price Data": price_data,
                "Fundamentals": fundamentals
            }

        if not results:
            logger.info(f"No stock data found for {company} on {date}")
            return f"No stock data found for {company} on {date}."

        logger.info(f"Stock price tool retrieved data for {company} on {date}")

        return {
            "Company": company,
            "Results": results
        }

    except Exception as e:
        logger.error(f"Error in stock_price_tool: {e}")
        return "Error retrieving stock data. Please ensure the query format is correct and try again."

from time import strptime
from ddgs import DDGS 
import yfinance as yf
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)

GNEWS_API_KEY="b56aee6c73868a137197b2e2db2b41a3"

# -------- HISTORICAL PRICE TOOL --------

#stock price retrieval tool
TICKER_MAP = {
    "HDFC": ["HDFCBANK.NS"],
    "RELIANCE": ["RELIANCE.NS"],
    "NIFTY": ["^NSEI"],
} 

def historical_price_tool(user_query: str):
    """
    Accepts:
    Date: YYYY-MM-DD
    Company: Name

    Returns price data for that specific date.
    """

    try:
        logger.info(f"Processing historical price query: {user_query}")

        lines = user_query.split("\n")
        date = lines[0].replace("Date:", "").strip()
        company = lines[1].replace("Company:", "").strip()

        company = company.strip().upper()  # Normalize company name

        tickers = TICKER_MAP.get(company)

        if not tickers:
            return f"No ticker mapping found for {company}."

        selected_date = datetime.strptime(date, "%Y-%m-%d")
        next_day = selected_date + timedelta(days=1)

        results = {}

        for ticker_symbol in tickers:

            stock = yf.Ticker(ticker_symbol)
            data = stock.history(start=selected_date, end=next_day)

            if data.empty:
                logger.info(f"No trading data for {ticker_symbol} on {date}")
                continue

            open_price = round(float(data["Open"].iloc[0]), 2)
            high_price = round(float(data["High"].iloc[0]), 2)
            low_price = round(float(data["Low"].iloc[0]), 2)
            close_price = round(float(data["Close"].iloc[0]), 2)
            volume = int(data["Volume"].iloc[0])

            pct_change = round(((close_price - open_price) / open_price) * 100, 2)
            intraday_range = round(high_price - low_price, 2)

            results[ticker_symbol] = {
                "Date": date,
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
                "Percentage Change (%)": pct_change,
                "Intraday Range": intraday_range
            }

        if not results:
            return f"No stock data found for {company} on {date}."

        logger.info(f"Historical tool retrieved data for {company} on {date}")

        return {
            "Company": company,
            "Results": results
        }

    except Exception as e:
        logger.error(f"Error in historical_price_tool: {e}")
        return "Error retrieving historical data."

# -------- FUNDAMENTAL TOOL --------

def fundamental_tool(user_query: str):
    """
    Accepts:
    Company: Name

    Returns company fundamental metrics.
    """

    try:
        logger.info(f"Processing fundamental query: {user_query}")

        lines = user_query.split("\n")
        company = lines[1].replace("Company:", "").strip()
        company = company.strip().upper()  # Normalize company name
        tickers = TICKER_MAP.get(company)

        if not tickers:
            return f"No ticker mapping found for {company}."

        results = {}

        for ticker_symbol in tickers:

            stock = yf.Ticker(ticker_symbol)
            info = stock.info

            fundamentals = {
                "Market Cap": info.get("marketCap"),
                "Shares Outstanding": info.get("sharesOutstanding"),
                "Trailing PE": info.get("trailingPE"),
                "Forward PE": info.get("forwardPE"),
                "Price to Book": info.get("priceToBook"),
                "Dividend Yield": info.get("dividendYield"),
                "Beta": info.get("beta"),
                "Return on Equity": info.get("returnOnEquity"),
                "Profit Margins": info.get("profitMargins"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry")
            }

            results[ticker_symbol] = fundamentals

        logger.info(f"Fundamental tool retrieved data for {company}")

        return {
            "Company": company,
            "Fundamentals": results
        }

    except Exception as e:
        logger.error(f"Error in fundamental_tool: {e}")
        return "Error retrieving fundamental data."

# -------- FINANCIAL STATEMENT TOOL --------

def financial_statement_tool(user_query: str):
    """
    Accepts:
    Company: Name

    Returns income statement, balance sheet, and cash flow.
    """

    try:
        logger.info(f"Processing financial statement query: {user_query}")

        lines = user_query.split("\n")
        company = lines[1].replace("Company:", "").strip()
        company = company.strip().upper()  # Normalize company name
        tickers = TICKER_MAP.get(company)

        if not tickers:
            return f"No ticker mapping found for {company}."

        results = {}

        for ticker_symbol in tickers:

            stock = yf.Ticker(ticker_symbol)

            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            results[ticker_symbol] = {
                "Income Statement": income_stmt.fillna(0).to_dict(),
                "Balance Sheet": balance_sheet.fillna(0).to_dict(),
                "Cash Flow": cash_flow.fillna(0).to_dict()
            }

        logger.info(f"Financial statement tool retrieved data for {company}")

        return {
            "Company": company,
            "Financial Statements": results
        }

    except Exception as e:
        logger.error(f"Error in financial_statement_tool: {e}")
        return "Error retrieving financial statements."



# news search with date filter
def get_gnews_articles(user_query: str):
    """
    Fetch news articles from GNews API with optional date filtering.
    
    Parameters:
        api_key (str): Your GNews API key
        query (str): Search keyword
        lang (str): Language code (default: "en")
        country (str): Country code (default: "in")
        from_date (str): Start date in YYYY-MM-DD format
        to_date (str): End date in YYYY-MM-DD format
        max_results (int): Number of articles (max 100)
        
    Returns:
        list: List of article dictionaries
    """

    try:
        logger.info(f"Fetching news articles for query: {user_query}")
        lines = user_query.split("\n")
        date = lines[0].replace("Date:", "").strip()
        company = lines[1].replace("Company:", "").strip()
        query = lines[2].replace("Question:", "").strip()

        date = date.replace(".", "-")

        logger.info(f"Parsed date: {date}, company: {company}")

        url = "https://gnews.io/api/v4/search"

        params = {
            "q": company,
            "lang": "en",
            "country": "in",
            "max": 20,
            "apikey": GNEWS_API_KEY
        }

        # Add date filters only if provided
        params["from"]= (pd.Timestamp(date) - timedelta(days=7)).strftime("%Y-%m-%d")  # 7 days before the given date
        params["to"] = date

        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article["title"],
                "description": article["description"],
                "url": article["url"],
                "published_at": article["publishedAt"],
                "source": article["source"]["name"]
            })

        logger.info(f"Fetched {len(articles)} articles from GNews API for company: {company} on date: {date}")

        return articles
    
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []




# if __name__ == "__main__":
#     print("Testing India Finance Search Tool. Type 'exit' to quit.")
#     while True:
#         q = "Date: 2026.02.03\nCompany: HDFCBANK.NS\nQuestion: what was the revenue on 2025?"
#         if q.lower() == "exit":
#             break
#         print("\nAnswer:", get_gnews_articles(q))




