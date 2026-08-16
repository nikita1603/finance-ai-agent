"""Quantitative risk/return analytics computed from price series.

Exposes `quant_analytics_tool(user_query: str)` which accepts the project's
structured query format and returns risk and performance metrics computed from
daily closing prices (via `yfinance`):

- Total and annualized (CAGR) return
- Annualized volatility and rolling 30-day volatility
- Sharpe and Sortino ratios (vs a constant risk-free rate)
- Maximum drawdown
- Market-model beta/alpha vs NIFTY 50 (^NSEI) estimated by OLS on daily
  excess returns, with R-squared and the standard error of beta

Date window resolution (in priority order):
  1. Explicit calendar years in the question ("in 2025", "2025 and 2026")
     → start = Jan 1 of earliest year, end = Dec 31 of latest year (capped at today)
  2. Relative period phrase ("last 6 months", "2 years")
     → lookback from the END of the selected Financial Year (March 31), capped at today
  3. Default: 1-year lookback ending on March 31 of the selected Financial Year

This means selecting "Financial Year: 2023-24" and asking "last year" correctly
returns data for April 2023 – March 2024, not the most recent 12 months.

Also exports `compute_market_beta` so other tools (e.g. fundamental_tool)
can reuse the OLS estimation logic without duplicating code.
"""

import logging
import re
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from backend.tools.utils import parse_structured_input, TICKER_MAP

logger = logging.getLogger(__name__)

RISK_FREE_RATE_ANNUAL = 0.065      # RBI repo-rate approximation
BENCHMARK_TICKER = "^NSEI"         # NIFTY 50
TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS = 60              # minimum trading days required


# ---------------------------------------------------------------------------
# Date range parsing
# ---------------------------------------------------------------------------

def _parse_financial_year_end(financial_year: str | None) -> datetime | None:
    """Return the last day (March 31) of a financial year string like '2023-24'.

    Returns None if the string is missing or cannot be parsed.
    """
    if not financial_year:
        return None
    m = re.match(r"(\d{4})-(\d{2,4})", financial_year.strip())
    if not m:
        return None
    start_year = int(m.group(1))
    suffix = m.group(2)
    end_year = (start_year // 100) * 100 + int(suffix) if len(suffix) == 2 else int(suffix)
    return datetime(end_year, 3, 31)


def _parse_date_range(question: str, reference_date: datetime) -> tuple[datetime, datetime]:
    """Return (start_date, end_date) inferred from the question and reference date.

    reference_date should be the end of the selected Financial Year (March 31),
    capped at today, so that relative phrases like "last year" or "last 6 months"
    are anchored to the correct FY window rather than always today.

    Priority order:
    1. Explicit calendar years in question ("2025", "2025 and 2026")
       → start = Jan 1 of earliest year, end = Dec 31 of latest year capped at today
       (explicit years override the financial year context)
    2. Relative period phrase ("last 6 months", "2 years")
       → lookback from reference_date
    3. Default: 1-year lookback from reference_date
    """
    today = datetime.today()
    # Cap reference_date at today so future dates don't produce empty ranges
    ref = min(reference_date, today)
    q = question.lower()

    # 1. Explicit 4-digit year mentions (e.g. "in 2025", "2025 and 2026")
    years_found = re.findall(r"\b(20[0-2]\d)\b", question)
    if years_found:
        years = sorted(set(int(y) for y in years_found))
        start_date = datetime(years[0], 1, 1)
        end_date = min(datetime(years[-1], 12, 31), today)
        return start_date, end_date

    # 2. Relative period phrases
    m = re.search(r"(\d+)\s*(?:year|yr)s?", q)
    if m:
        return ref - timedelta(days=int(m.group(1)) * 365), ref
    m = re.search(r"(\d+)\s*(?:month|mo)s?", q)
    if m:
        return ref - timedelta(days=int(m.group(1)) * 30), ref
    m = re.search(r"(\d+)\s*(?:week|wk)s?", q)
    if m:
        return ref - timedelta(days=int(m.group(1)) * 7), ref

    # 3. Default: 1 year lookback from reference date
    return ref - timedelta(days=365), ref


# ---------------------------------------------------------------------------
# OLS market-model estimation
# ---------------------------------------------------------------------------

def compute_market_beta(
    stock_returns: pd.Series,
    bench_returns: pd.Series,
    rf_daily: float,
) -> dict | None:
    """Estimate market-model beta via OLS on daily excess returns.

    Regresses (stock - rf) on (benchmark - rf) using closed-form OLS.
    Returns None when fewer than MIN_OBSERVATIONS aligned observations exist.

    Returns a dict with keys:
        beta, alpha_annualized, r_squared, beta_std_error, observations
    """
    excess_stock = stock_returns - rf_daily
    excess_bench = bench_returns - rf_daily

    aligned = pd.concat([excess_stock, excess_bench], axis=1).dropna()
    if len(aligned) < MIN_OBSERVATIONS:
        return None

    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    n = len(y)

    x_mean, y_mean = x.mean(), y.mean()
    ss_xx = float(np.sum((x - x_mean) ** 2))
    if ss_xx == 0:
        return None

    beta = float(np.sum((x - x_mean) * (y - y_mean)) / ss_xx)
    alpha_daily = y_mean - beta * x_mean
    alpha_annual = alpha_daily * TRADING_DAYS_PER_YEAR

    residuals = y - (alpha_daily + beta * x)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    se_beta = float(np.sqrt(ss_res / max(n - 2, 1)) / np.sqrt(ss_xx))

    return {
        "beta": round(float(beta), 4),
        "alpha_annualized": round(float(alpha_annual), 6),
        "r_squared": round(float(r_squared), 4),
        "beta_std_error": round(float(se_beta), 4),
        "observations": n,
    }


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------

def quant_analytics_tool(user_query: str) -> dict | str:
    """Compute risk/return metrics for a company from historical daily prices.

    Date window is resolved in this priority order:
      1. Explicit years in the question ("in 2025", "2025 and 2026") override everything.
      2. Relative phrases ("last 6 months", "2 years") are applied relative to the
         end of the selected Financial Year (March 31), capped at today.
      3. No period specified → defaults to the full Financial Year window
         (1 year ending March 31 of the selected FY).

    Returns a dict with keys:
        company, ticker, period_days, start_date, end_date, trading_days,
        total_return, cagr, annualized_volatility, rolling_30d_volatility,
        sharpe_ratio, sortino_ratio, max_drawdown, risk_free_rate_annual,
        benchmark, market_model

    On failure returns an error string.
    """
    try:
        data = parse_structured_input(user_query)
        company = data["company"]
        question = data.get("question") or ""

        tickers = TICKER_MAP.get(company)
        if not tickers:
            return f"No ticker mapping found for {company}."

        ticker_symbol = tickers[0]
        today = datetime.today()

        # Reference date priority:
        # 1. End of selected financial year (March 31), capped at today
        #    → anchors "last year" / default lookback to the correct FY window
        # 2. Date: field from structured input, capped at today
        # 3. Today
        financial_year = data.get("financial_year")
        date_str = data.get("date")

        fy_end = _parse_financial_year_end(financial_year)
        if fy_end:
            reference_date = min(fy_end, today)
        elif date_str:
            try:
                reference_date = min(datetime.strptime(date_str, "%Y-%m-%d"), today)
            except ValueError:
                reference_date = today
        else:
            reference_date = today

        start_date, end_date = _parse_date_range(question, reference_date)
        period_days = (end_date - start_date).days

        logger.info(
            "Fetching %s price data from %s to %s",
            ticker_symbol, start_date.date(), end_date.date(),
        )

        stock_df = yf.download(
            ticker_symbol, start=start_date, end=end_date,
            progress=False, auto_adjust=True,
        )
        bench_df = yf.download(
            BENCHMARK_TICKER, start=start_date, end=end_date,
            progress=False, auto_adjust=True,
        )

        if stock_df.empty:
            return f"No price data returned for {company} ({ticker_symbol})."

        close = stock_df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        close.index = close.index.tz_localize(None)

        if len(close) < MIN_OBSERVATIONS:
            return (
                f"Insufficient data for {company}: {len(close)} trading days "
                f"(need ≥ {MIN_OBSERVATIONS})."
            )

        daily_returns = close.pct_change().dropna()
        n = len(daily_returns)
        rf_daily = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR

        # ── Core return/risk metrics ────────────────────────────────────────
        total_return = float(close.iloc[-1] / close.iloc[0] - 1)
        years = period_days / 365.0
        cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else float("nan")

        ann_vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        excess_daily = daily_returns - rf_daily
        ann_excess = float(excess_daily.mean() * TRADING_DAYS_PER_YEAR)
        sharpe = ann_excess / ann_vol if ann_vol > 0 else float("nan")

        downside = daily_returns[daily_returns < rf_daily]
        downside_vol = float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside) > 1 else float("nan")
        sortino = ann_excess / downside_vol if downside_vol > 0 else float("nan")

        rolling_peak = close.cummax()
        drawdown = (close - rolling_peak) / rolling_peak
        max_drawdown = float(drawdown.min())

        rolling_vol_series = daily_returns.rolling(30).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        rolling_vol_latest = float(rolling_vol_series.iloc[-1]) if not rolling_vol_series.empty else None

        # ── Market model ────────────────────────────────────────────────────
        market_model = None
        if not bench_df.empty:
            bench_close = bench_df["Close"].squeeze()
            if isinstance(bench_close, pd.DataFrame):
                bench_close = bench_close.iloc[:, 0]
            bench_close = bench_close.dropna()
            bench_close.index = bench_close.index.tz_localize(None)
            bench_returns = bench_close.pct_change().dropna()
            market_model = compute_market_beta(daily_returns, bench_returns, rf_daily)

        result = {
            "company": company,
            "ticker": ticker_symbol,
            "period_days": period_days,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "trading_days": n,
            "total_return": round(total_return, 4),
            "cagr": round(cagr, 4) if not np.isnan(cagr) else None,
            "annualized_volatility": round(ann_vol, 4),
            "rolling_30d_volatility": round(rolling_vol_latest, 4) if rolling_vol_latest and not np.isnan(rolling_vol_latest) else None,
            "sharpe_ratio": round(sharpe, 4) if not np.isnan(sharpe) else None,
            "sortino_ratio": round(sortino, 4) if not np.isnan(sortino) else None,
            "max_drawdown": round(max_drawdown, 4),
            "risk_free_rate_annual": RISK_FREE_RATE_ANNUAL,
            "benchmark": f"NIFTY 50 ({BENCHMARK_TICKER})",
            "market_model": market_model,
        }

        logger.info("Quant analytics result for %s: %s", company, result)
        return result

    except Exception as e:
        logger.error("quant_analytics_tool failed: %s", e)
        return f"Error computing quantitative analytics: {e}"
