"""Fetch company fundamental metrics using `yfinance`.

This module exposes `fundamental_tool(user_query: str)` which accepts a
structured query (parsed by `parse_structured_input`) and returns a dict
with basic fundamental metrics for the mapped tickers or an error string.

"""

import logging
import yfinance as yf
from backend.tools.utils import parse_structured_input, TICKER_MAP

logger = logging.getLogger(__name__)


def fundamental_tool(user_query: str):
    """Retrieve fundamental company metrics for mapped ticker(s).

    The function expects `user_query` to contain a `Company:` field that
    matches a key in `TICKER_MAP`. It returns a dictionary containing the
    company name and a `Fundamentals` mapping of ticker -> metrics.

    On error the function returns a string describing the failure.
    """
    try:
        # Parse structured input to obtain the company identifier
        data = parse_structured_input(user_query)

        logger.info(f"Parsed structured input for fundamental tool: {data}")

        company = data["company"]

        # Resolve company to one or more ticker symbols
        tickers = TICKER_MAP.get(company)
        if not tickers:
            return f"No ticker mapping found for {company}."

        logger.info(f"Retrieving fundamental data for {company} using tickers: {tickers}")

        results = {}

        # Fetch quote/info for each ticker and extract chosen fields
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
        # Preserve original error handling while logging the exception
        logger.error(e)
        return f"Error retrieving fundamental data: {e}"