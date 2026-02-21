import logging
import requests
import yfinance as yf
import pandas as pd

from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

GNEWS_API_KEY="b56aee6c73868a137197b2e2db2b41a3"

# ==============================================================
# ---------------- TICKER MAP ----------------
# ==============================================================

TICKER_MAP = {
    "HDFC": ["HDFCBANK.NS"],
    "RELIANCE": ["RELIANCE.NS"],
    "NIFTY": ["^NSEI"],
}

# ==============================================================
# ---------------- STRUCTURED INPUT PARSER ----------------
# ==============================================================

def parse_structured_input(user_query: str):

    parsed = {
        "date": None,
        "company": None,
        "financial_year": None,
        "quarter": None,
        "question": None
    }

    for line in user_query.split("\n"):
        if line.startswith("Date:"):
            parsed["date"] = line.replace("Date:", "").strip()
        elif line.startswith("Company:"):
            parsed["company"] = line.replace("Company:", "").strip().upper()
        elif line.startswith("Financial Year:"):
            parsed["financial_year"] = line.replace("Financial Year:", "").strip()
        elif line.startswith("Quarter:"):
            parsed["quarter"] = line.replace("Quarter:", "").strip()
        elif line.startswith("Question:"):
            parsed["question"] = line.replace("Question:", "").strip()

    return parsed


# ==============================================================
# ---------------- HISTORICAL PRICE TOOL ----------------
# ==============================================================

def historical_price_tool(user_query: str):

    try:
        data = parse_structured_input(user_query)

        logger.info(f"Parsed structured input: {data}")

        company = data["company"]
        date_str = data["date"]

        tickers = TICKER_MAP.get(company)
        if not tickers:
            return f"No ticker mapping found for {company}."

        requested_date = datetime.strptime(date_str, "%Y-%m-%d")

        logger.info(f"Retrieving historical price for {company} on {date_str} using tickers: {tickers}")

        results = {}

        for ticker_symbol in tickers:

            stock = yf.Ticker(ticker_symbol)

            # Try up to 7 days backward to find trading day
            trading_date = requested_date
            max_lookback = 7
            found = False

            logger.info(f"Attempting to retrieve data for {ticker_symbol} starting from {trading_date.strftime('%Y-%m-%d')}")

            for _ in range(max_lookback):
                next_day = trading_date + timedelta(days=1)
                df = stock.history(start=trading_date, end=next_day)

                if not df.empty:
                    found = True
                    break

                trading_date -= timedelta(days=1)

            if not found:
                return f"No trading data found within {max_lookback} days before {date_str}."
            
            logger.info(f"Data retrieved for {ticker_symbol} on {trading_date.strftime('%Y-%m-%d')}")

            open_price = float(df["Open"].iloc[0])
            close_price = float(df["Close"].iloc[0])
            high_price = float(df["High"].iloc[0])
            low_price = float(df["Low"].iloc[0])
            volume = int(df["Volume"].iloc[0])

            pct_change = round(((close_price - open_price) / open_price) * 100, 2)

            results[ticker_symbol] = {
                "Requested Date": date_str,
                "Actual Trading Date Used": trading_date.strftime("%Y-%m-%d"),
                "Open": round(open_price, 2),
                "High": round(high_price, 2),
                "Low": round(low_price, 2),
                "Close": round(close_price, 2),
                "Volume": volume,
                "Percentage Change (%)": pct_change
            }

        logger.info(f"Historical price data compiled for {company}: {results}")

        return {
            "Company": company,
            "Price Data": results
        }

    except Exception as e:
        logger.error(e)
        return f"Error retrieving historical data: {e}"


# ==============================================================
# ---------------- FUNDAMENTAL TOOL ----------------
# ==============================================================

def fundamental_tool(user_query: str):

    try:
        data = parse_structured_input(user_query)

        logger.info(f"Parsed structured input for fundamental tool: {data}")

        company = data["company"]

        tickers = TICKER_MAP.get(company)
        if not tickers:
            return f"No ticker mapping found for {company}."
        

        logger.info(f"Retrieving fundamental data for {company} using tickers: {tickers}")

        results = {}

        for ticker_symbol in tickers:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info

            fundamentals = {
                "Market Cap": info.get("marketCap"),
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

        logger.info(f"Fundamental data compiled for {company}: {results}")

        return {
            "Company": company,
            "Fundamentals": results
        }

    except Exception as e:
        logger.error(e)
        return f"Error retrieving fundamental data: {e}"


# ==============================================================
# ---------------- FINANCIAL STATEMENT TOOL ----------------
# ==============================================================

def financial_statement_tool(user_query: str):

    try:
        data = parse_structured_input(user_query)

        logger.info(f"Parsed structured input for financial statement tool: {data}")

        company = data["company"]
        fy = data["financial_year"]
        quarter = data["quarter"]

        tickers = TICKER_MAP.get(company)
        if not tickers:
            return f"No ticker mapping found for {company}."

        if not fy:
            return "Financial Year must be provided in format YYYY-YY."

        # Extract FY end year (Indian FY: 2025-26 → 2026)
        try:
            fy_start = int(fy.split("-")[0])
            fy_end = fy_start + 1
        except:
            return "Invalid Financial Year format. Use YYYY-YY."

        quarter_month_map = {
            "Q1": (6, 30),   # June 30
            "Q2": (9, 30),   # Sept 30
            "Q3": (12, 31),  # Dec 31
            "Q4": (3, 31)    # March 31 (FY end)
        }

        logger.info(f"Retrieving financial statements for {company}, FY: {fy}, Quarter: {quarter} using tickers: {tickers}")

        results = {}

        for ticker_symbol in tickers:

            stock = yf.Ticker(ticker_symbol)

            # -------- Quarterly --------
            if quarter and quarter != "None":

                logger.info(f"Processing {quarter} financial statement for {ticker_symbol}")

                if quarter not in quarter_month_map:
                    return "Invalid quarter value. Use Q1, Q2, Q3, or Q4."

                month, day = quarter_month_map[quarter]

                if quarter == "Q4":
                    target_year = fy_end
                else:
                    target_year = fy_start

                target_date = pd.Timestamp(year=target_year, month=month, day=day)

                df = stock.quarterly_financials

                if df.empty:
                    return "Quarterly financial data unavailable."

                if target_date not in df.columns:
                    return f"No financial data found for {quarter} FY {fy}."

                statement = df[target_date]

            # -------- Annual --------
            else:

                logger.info(f"Processing annual financial statement for {ticker_symbol}")

                target_date = pd.Timestamp(year=fy_end, month=3, day=31)

                df = stock.financials

                if df.empty:
                    return "Annual financial data unavailable."

                if target_date not in df.columns:
                    return f"No annual financial data found for FY {fy}."

                statement = df[target_date]

            results[ticker_symbol] = statement.fillna(0).to_dict()

        logger.info(f"Financial statement data compiled for {company}: {results}")

        return {
            "Company": company,
            "Financial Year": fy,
            "Quarter": quarter,
            "Financial Data": results
        }

    except Exception as e:
        logger.error(e)
        return f"Error retrieving financial statements: {e}"


# ==============================================================
# ---------------- NEWS TOOL ----------------
# ==============================================================

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