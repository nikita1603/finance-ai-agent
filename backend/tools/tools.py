"""Tool registry for the agent.

This module defines `TOOLS`, a list of `FunctionTool` wrappers that expose
project-specific helper functions (RAG search, news, historical prices,
and fundamentals) to the `FunctionAgent`.
"""

from llama_index.core.tools import FunctionTool
from backend.tools.company_financial_statement_tool.rag_model import rag_tool
from backend.tools.historical_price_tool.yfinance_tool import historical_price_tool
from backend.tools.company_fundamental_tool.yfinance_tools import fundamental_tool
from backend.tools.news_tool.gnews_tool import get_gnews_articles

# List of tools exposed to the agent. Each entry is a FunctionTool that
# provides a name, a callable, and a description used by the agent's
# tool-selection logic.
TOOLS = [

    # RAG tool: document-grounded financial numbers + qualitative analysis
    FunctionTool.from_defaults(
        fn=rag_tool,
        name="rag_tool",
        description=(
            "Primary tool for retrieving information from indexed company documents "
            "(earnings call transcripts, investor presentations, annual reports, quarterly results PDFs).\n\n"

            "USE THIS TOOL FOR:\n"
            "- Exact financial figures: revenue, net interest income, PAT, NIM, EPS, GNPA, NNPA, "
            "capital adequacy ratio, deposits, advances, and any other line items reported in quarterly/annual filings.\n"
            "- Qualitative analysis: management commentary, strategic guidance, business highlights, "
            "margin drivers, segment performance, and outlook.\n"
            "- Period-specific questions: any question tied to a specific quarter (Q1/Q2/Q3/Q4) or financial year.\n\n"

            "INPUT FORMAT (pass the full structured block exactly as received):\n"
            "Date: YYYY-MM-DD\n"
            "Company: Company Name\n"
            "Financial Year: YYYY-YY\n"
            "Quarter: Q1/Q2/Q3/Q4/None\n"
            "Question: User Question\n\n"

            "Example questions this tool handles:\n"
            "- What was the net interest income in Q3 FY2025-26?\n"
            "- What is the GNPA ratio for Q4 FY2025-26?\n"
            "- Summarize management commentary from the Q2 FY2025-26 earnings call.\n"
            "- Why did margins decline in Q3 FY2025-26?\n"
            "- What was the PAT for FY2024-25?\n\n"

            "DO NOT use this tool for:\n"
            "- Current/live stock price or intraday price data\n"
            "- Live valuation ratios (P/E, P/B — use fundamental_tool)\n"
            "- Recent market news or stock movement catalysts (use get_gnews_articles)\n"
        )
    ),

    # News tool: event-driven analysis via GNews
    FunctionTool.from_defaults(
        fn=get_gnews_articles,
        name="get_gnews_articles",
        description=(
            "Fetches recent news articles for a company from GNews.\n\n"

            "USE THIS TOOL FOR:\n"
            "- Reasons behind stock price movement on a specific date\n"
            "- Recent regulatory updates, RBI actions, or policy changes affecting the company\n"
            "- Earnings reactions, analyst upgrades/downgrades, or target price changes\n"
            "- Macro events, sector news, or any external catalyst affecting the stock\n\n"

            "INPUT FORMAT (pass the full structured block exactly as received):\n"
            "Date: YYYY-MM-DD\n"
            "Company: Company Name\n"
            "Financial Year: YYYY-YY\n"
            "Quarter: Q1/Q2/Q3/Q4/None\n"
            "Question: User Question\n\n"

            "OUTPUT INSTRUCTIONS:\n"
            "- Return headline + 1-2 line summary + article URL for each relevant article.\n"
            "- Do NOT paste full article body.\n"
            "- Only include articles directly relevant to the company or its market context.\n\n"

            "DO NOT use this tool for:\n"
            "- Financial statement numbers\n"
            "- Valuation ratios\n"
            "- Historical stock prices\n"
        )
    ),

    # Historical price tool: OHLCV for a specific date
    FunctionTool.from_defaults(
        fn=historical_price_tool,
        name="historical_price_tool",
        description=(
            "Retrieves historical OHLCV (Open, High, Low, Close, Volume) stock price data "
            "for a company on a specific date.\n\n"

            "USE THIS TOOL FOR:\n"
            "- Stock price on a specific date\n"
            "- Price movement or trading volume on a particular trading day\n\n"

            "INPUT FORMAT (pass the full structured block exactly as received):\n"
            "Date: YYYY-MM-DD\n"
            "Company: Company Name\n"
            "Financial Year: YYYY-YY\n"
            "Quarter: Q1/Q2/Q3/Q4/None\n"
            "Question: User Question\n\n"

            "Note: If the requested date is not a trading day, the tool looks back up to 7 days "
            "to find the nearest available trading session.\n\n"

            "DO NOT use this tool for:\n"
            "- Financial statement numbers (revenue, PAT, NIM, etc.)\n"
            "- Valuation ratios\n"
            "- Company performance commentary\n"
            "- News or event analysis\n"
        )
    ),

    # Fundamental tool: current valuation ratios and company profile
    FunctionTool.from_defaults(
        fn=fundamental_tool,
        name="fundamental_tool",
        description=(
            "Retrieves current/latest company valuation ratios and fundamental metrics from market data.\n\n"

            "USE THIS TOOL FOR:\n"
            "- Valuation: Market Cap, P/E (Trailing and Forward), Price-to-Book\n"
            "- Returns: ROE, ROA, Profit Margins\n"
            "- Income: Dividend Yield\n"
            "- Risk: Beta\n"
            "- Profile: Sector, Industry\n\n"

            "INPUT FORMAT (pass the full structured block exactly as received):\n"
            "Date: YYYY-MM-DD\n"
            "Company: Company Name\n"
            "Financial Year: YYYY-YY\n"
            "Quarter: Q1/Q2/Q3/Q4/None\n"
            "Question: User Question\n\n"

            "Note: This tool returns CURRENT/LATEST market data, not historical period data.\n\n"

            "DO NOT use this tool for:\n"
            "- Historical revenue, PAT, or income statement figures\n"
            "- Period-specific financial results (use rag_tool)\n"
            "- Historical stock prices (use historical_price_tool)\n"
            "- News or market catalysts (use get_gnews_articles)\n"
        )
    ),
]