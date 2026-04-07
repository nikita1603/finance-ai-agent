from dotenv import load_dotenv
import os

load_dotenv()

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

TICKER_MAP = {
    "HDFC": ["HDFCBANK.NS"],
    "RELIANCE": ["RELIANCE.NS"],
    "NIFTY": ["^NSEI"],
}

def parse_structured_input(user_query: str):

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