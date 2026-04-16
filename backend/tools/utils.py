"""Utility helpers for parsing structured queries and shared constants.

This module provides a small parser `parse_structured_input` used across
tools to extract fields from the project's structured query format, plus
shared constants such as `GNEWS_API_KEY` and `TICKER_MAP`.

The structured input format expected by `parse_structured_input` is:
    Date: YYYY-MM-DD
    Company: Company Name
    Financial Year: YYYY-YY
    Quarter: Q1/Q2/Q3/Q4/None
    Question: User Question

"""

from dotenv import load_dotenv
import os

# Load environment variables from .env if present
load_dotenv()

# API key for GNews (used by the news tool)
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

# Simple mapping from normalized company identifier to one or more
# ticker symbols used with yfinance. Keys are expected to be uppercase.
TICKER_MAP = {
    "HDFC": ["HDFCBANK.NS"],
    "RELIANCE": ["RELIANCE.NS"],
    "NIFTY": ["^NSEI"],
}


def parse_structured_input(user_query: str):
    """Parse a structured, multi-line user query into a dict.

    The parser reads each line, strips leading/trailing whitespace and
    extracts values for the recognized prefixes. Returned keys are:
    `date`, `company`, `financial_year`, `quarter`, `question`.

    The `company` value is normalized to uppercase to match keys in
    `TICKER_MAP`.
    """

    parsed = {
        "date": None,
        "company": None,
        "financial_year": None,
        "quarter": None,
        "question": None
    }

    for line in user_query.split("\n"):
        line = line.strip()
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