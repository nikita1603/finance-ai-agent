import logging
import yfinance as yf
import pandas as pd
from backend.tools.utils import parse_structured_input, TICKER_MAP

logger = logging.getLogger(__name__)

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
