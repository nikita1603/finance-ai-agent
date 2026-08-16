"""Fetch company fundamental metrics using `yfinance`.

Exposes `fundamental_tool(user_query: str)` which returns current valuation
ratios plus two beta estimates:

- Published beta: yfinance's 5-year monthly beta sourced from Yahoo Finance.
- Computed beta: 1-year daily OLS beta vs NIFTY 50 (^NSEI) calculated via
  `compute_market_beta` from the quant analytics module, making the estimation
  methodology difference explicit to the LLM and the end user.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from backend.tools.utils import parse_structured_input, TICKER_MAP
from backend.tools.quant_analytics_tool.quant_tool import (
    compute_market_beta,
    RISK_FREE_RATE_ANNUAL,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)

_BETA_LOOKBACK_DAYS = 365


def _compute_1y_daily_beta(ticker_symbol: str) -> dict | None:
    """Return OLS beta dict for ticker vs NIFTY 50 over the last year.

    Returns None on any failure so the caller can degrade gracefully.
    """
    try:
        end = datetime.today()
        start = end - timedelta(days=_BETA_LOOKBACK_DAYS)
        stock_df = yf.download(ticker_symbol, start=start, end=end, progress=False, auto_adjust=True)
        bench_df = yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True)

        if stock_df.empty or bench_df.empty:
            return None

        stock_close = stock_df["Close"].squeeze()
        if isinstance(stock_close, pd.DataFrame):
            stock_close = stock_close.iloc[:, 0]
        stock_close = stock_close.dropna()
        stock_close.index = stock_close.index.tz_localize(None)

        bench_close = bench_df["Close"].squeeze()
        if isinstance(bench_close, pd.DataFrame):
            bench_close = bench_close.iloc[:, 0]
        bench_close = bench_close.dropna()
        bench_close.index = bench_close.index.tz_localize(None)

        rf_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR
        return compute_market_beta(
            stock_close.pct_change().dropna(),
            bench_close.pct_change().dropna(),
            rf_daily,
        )
    except Exception as e:
        logger.warning("Could not compute 1Y daily beta for %s: %s", ticker_symbol, e)
        return None


def fundamental_tool(user_query: str):
    """Retrieve fundamental company metrics for mapped ticker(s).

    Returns a dict with 'Company' and 'Fundamentals' (per-ticker metrics).
    The beta field includes both the published 5Y monthly value from Yahoo
    Finance and a freshly-computed 1Y daily OLS estimate vs NIFTY 50.

    On error returns a string describing the failure.
    """
    try:
        data = parse_structured_input(user_query)
        logger.info("Parsed structured input for fundamental tool: %s", data)

        company = data["company"]
        tickers = TICKER_MAP.get(company)
        if not tickers:
            return f"No ticker mapping found for {company}."

        logger.info("Retrieving fundamental data for %s using tickers: %s", company, tickers)

        results = {}
        for ticker_symbol in tickers:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info

            computed_beta = _compute_1y_daily_beta(ticker_symbol)

            fundamentals = {
                "Market Cap": info.get("marketCap"),
                "Trailing PE": info.get("trailingPE"),
                "Forward PE": info.get("forwardPE"),
                "Price to Book": info.get("priceToBook"),
                "Dividend Yield": info.get("dividendYield"),
                "Beta (Published 5Y Monthly)": info.get("beta"),
                "Beta (Computed 1Y Daily OLS vs NIFTY 50)": computed_beta,
                "Return on Equity": info.get("returnOnEquity"),
                "Profit Margins": info.get("profitMargins"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
            }

            results[ticker_symbol] = fundamentals

        logger.info("Fundamental data compiled for %s: %s", company, results)
        return {"Company": company, "Fundamentals": results}

    except Exception as e:
        logger.error(e)
        return f"Error retrieving fundamental data: {e}"
