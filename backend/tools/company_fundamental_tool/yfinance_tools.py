import logging
import yfinance as yf
from backend.tools.utils import parse_structured_input, TICKER_MAP

logger = logging.getLogger(__name__)

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