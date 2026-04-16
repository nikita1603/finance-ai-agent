"""Historical price tool using `yfinance`.

This module provides `historical_price_tool(user_query: str)` which expects a
structured input (parsed by `parse_structured_input`) containing a `Company:`
and `Date:` field. It attempts to fetch the trading data for the requested
date, looking back up to a configurable window if the requested date was not
a trading day (e.g., weekend or holiday).

"""

import logging
import yfinance as yf
from datetime import datetime, timedelta
from backend.tools.utils import parse_structured_input, TICKER_MAP

logger = logging.getLogger(__name__)


def historical_price_tool(user_query: str):
    """Retrieve historical price data for a company on a requested date.

    The function expects `user_query` to include a `Company:` and `Date:`
    (YYYY-MM-DD). It returns a dictionary mapping tickers to a small set of
    price fields. If the requested date is a non-trading day, the function
    looks backward up to 7 days to find the most recent trading day.

    Returns a dict with `Company` and `Price Data`, or an error string.
    """

    try:
        # Parse structured input (company, date, etc.)
        data = parse_structured_input(user_query)

        logger.info(f"Parsed structured input: {data}")

        company = data["company"]
        date_str = data["date"]

        # Resolve company -> list of ticker symbols
        tickers = TICKER_MAP.get(company)
        if not tickers:
            return f"No ticker mapping found for {company}."

        # Parse requested date string into datetime
        requested_date = datetime.strptime(date_str, "%Y-%m-%d")

        logger.info(f"Retrieving historical price for {company} on {date_str} using tickers: {tickers}")

        results = {}

        for ticker_symbol in tickers:

            stock = yf.Ticker(ticker_symbol)

            # Try up to 7 days backward to find the latest trading day
            trading_date = requested_date
            max_lookback = 7
            found = False

            logger.info(f"Attempting to retrieve data for {ticker_symbol} starting from {trading_date.strftime('%Y-%m-%d')}")

            for _ in range(max_lookback):
                # Fetch one-day window [trading_date, next_day)
                next_day = trading_date + timedelta(days=1)
                df = stock.history(start=trading_date, end=next_day)

                if not df.empty:
                    found = True
                    break

                # Move back one day and retry
                trading_date -= timedelta(days=1)

            if not found:
                return f"No trading data found within {max_lookback} days before {date_str}."
            
            logger.info(f"Data retrieved for {ticker_symbol} on {trading_date.strftime('%Y-%m-%d')}")

            # Extract primary OHLCV fields and compute percentage change
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