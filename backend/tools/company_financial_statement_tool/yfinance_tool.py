"""Tool to fetch company financial statements using yfinance.

This module exposes `financial_statement_tool(user_query: str)` which expects
structured input (see `parse_structured_input`) and returns either a dict
containing requested financial data or an error message string.

"""

import logging
import yfinance as yf
import pandas as pd
from backend.tools.utils import parse_structured_input, TICKER_MAP

logger = logging.getLogger(__name__)


def financial_statement_tool(user_query: str):
    """Retrieve financial statements for a company using `yfinance`.

    The function expects `user_query` to be a structured multi-line string
    (fields parsed by `parse_structured_input`). It supports either
    quarter-level retrieval (Q1..Q4) or full fiscal-year (annual) retrieval.

    Returns:
        - dict with keys `Company`, `Financial Year`, `Quarter`, `Financial Data` on success
        - string error message on failure
    """

    try:
        # Extract structured fields from user input
        data = parse_structured_input(user_query)

        logger.info(f"Parsed structured input for financial statement tool: {data}")

        company = data["company"]
        fy = data["financial_year"]
        quarter = data["quarter"]

        # Map company name to one or more ticker symbols
        tickers = TICKER_MAP.get(company)
        if not tickers:
            return f"No ticker mapping found for {company}."

        if not fy:
            return "Financial Year must be provided in format YYYY-YY."

        # Extract FY start/end years. For Indian FY notation like '2025-26'
        # we take the first part as the start year and the next year as end.
        try:
            fy_start = int(fy.split("-")[0])
            fy_end = fy_start + 1
        except:
            return "Invalid Financial Year format. Use YYYY-YY."

        # Mapping from quarter code to month/day used for target timestamps
        quarter_month_map = {
            "Q1": (6, 30),   # June 30
            "Q2": (9, 30),   # Sept 30
            "Q3": (12, 31),  # Dec 31
            "Q4": (3, 31)    # March 31 (FY end)
        }

        logger.info(f"Retrieving financial statements for {company}, FY: {fy}, Quarter: {quarter} using tickers: {tickers}")

        results = {}

        # Iterate over mapped tickers and fetch financials
        for ticker_symbol in tickers:

            stock = yf.Ticker(ticker_symbol)

            # -------- Quarterly retrieval --------
            if quarter and quarter != "None":

                logger.info(f"Processing {quarter} financial statement for {ticker_symbol}")

                if quarter not in quarter_month_map:
                    return "Invalid quarter value. Use Q1, Q2, Q3, or Q4."

                month, day = quarter_month_map[quarter]

                # Q4 corresponds to the FY end year, other quarters map to FY start
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

            # -------- Annual retrieval --------
            else:

                logger.info(f"Processing annual financial statement for {ticker_symbol}")

                # Annual statements are tied to FY end (March 31)
                target_date = pd.Timestamp(year=fy_end, month=3, day=31)

                df = stock.financials

                if df.empty:
                    return "Annual financial data unavailable."

                if target_date not in df.columns:
                    return f"No annual financial data found for FY {fy}."

                statement = df[target_date]

            # Convert pandas Series to dict and replace NaNs with 0
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
