from llama_index.core.tools import FunctionTool
from backend.tools.company_financial_statement_tool.rag_model import rag_tool
from backend.tools.company_financial_statement_tool.yfinance_tool import financial_statement_tool
from backend.tools.historical_price_tool.yfinance_tool import historical_price_tool
from backend.tools.company_fundamental_tool.yfinance_tools import fundamental_tool
from backend.tools.news_tool.gnews_tool import get_gnews_articles

TOOLS = [

    FunctionTool.from_defaults(
        fn=rag_tool,
        name="rag_tool",
        description=(
            "Use this tool for qualitative, narrative, or explanatory analysis "
            "from company documents such as earnings calls, investor presentations, "
            "annual reports, and quarterly reports.\n\n"

            "This tool can ALSO be used for getting financial numbers for period of time whose data is available in database.\n"

            "INPUT SHOULD BE IN THE FOLLOWING STRUCTURED FORMAT:\n"
            " Date: YYYY-MM-DD\nCompany: Company Name\nFinancial Year: YYYY-YY\nQuarter: Q1/Q2/Q3/Q4/None\nQuestion: User Question\n\n"

            "Examples:\n"
            "- Explain Q3 2025-26 performance.\n"
            "- What were the key highlights in Q1 2025-26?\n"
            "- Summarize management commentary for Q4 2025-26.\n"
            "- Why did margins decline in Q3 2025-26?\n"
            "- Headline from Q2 2025-26 earnings call.\n\n"

            "Do NOT use this tool for:\n"
            "- Exact numeric values (revenue, EPS, net income, margins)\n"
            "- Stock price movement\n"
            "- Valuation ratios\n"
            "- Market news\n"
        )
    ),

    FunctionTool.from_defaults(
        fn=get_gnews_articles,
        name="get_gnews_articles",
        description=(
            "Use this tool when the user asks about reasons for stock movement, recent news, "
            "announcements, earnings reactions, regulatory updates, macro events, "
            "or any event-driven question affecting the stock or market.\n\n"

            "INPUT SHOULD BE IN THE FOLLOWING STRUCTURED FORMAT:\n"
            " Date: YYYY-MM-DD\nCompany: Company Name\nFinancial Year: YYYY-YY\nQuarter: Q1/Q2/Q3/Q4/None\nQuestion: User Question\n\n"

            "Use this tool for sentiment and catalyst analysis.\n\n"
            "Do NOT paste full article content. Instead:\n"
            "- Extract headline\n"
            "- Provide a concise 1-2 line summary\n"
            "- Include the article hyperlink\n\n"
            "Only include news relevant to the company or broader market context."
        )
    ),

    FunctionTool.from_defaults(
        fn=historical_price_tool,
        name="historical_price_tool",
        description=(
            "Use this tool ONLY for historical stock price data for a specific date.\n\n"

            "INPUT SHOULD BE IN THE FOLLOWING STRUCTURED FORMAT:\n"
            " Date: YYYY-MM-DD\nCompany: Company Name\nFinancial Year: YYYY-YY\nQuarter: Q1/Q2/Q3/Q4/None\nQuestion: User Question\n\n"

            "Examples:\n"
            "- What was the stock price on 2026-01-15?\n"
            "- Price movement on a specific trading day.\n\n"

            "This tool retrieves OHLCV data (Open, High, Low, Close, Volume) for a specific date. "
            "If the exact date is not a trading day, it looks back up to 7 days to find the nearest trading day.\n\n"

            "Do NOT use this tool for:\n"
            "- Financial statement numbers\n"
            "- Company performance discussion\n"
            "- News analysis\n"
        )
    ),

    FunctionTool.from_defaults(
        fn=fundamental_tool,
        name="fundamental_tool",
        description=(
            "Use this tool ONLY for company valuation and fundamental ratios (latest/current data).\n\n"

            "INPUT SHOULD BE IN THE FOLLOWING STRUCTURED FORMAT:\n"
            " Date: YYYY-MM-DD\nCompany: Company Name\nFinancial Year: YYYY-YY\nQuarter: Q1/Q2/Q3/Q4/None\nQuestion: User Question\n\n"

            "Returns: Market Cap, P/E ratios (Trailing and Forward), Price to Book, Dividend Yield, "
            "Beta, ROE, Profit Margins, Sector, and Industry.\n\n"

            "Examples:\n"
            "- What is the P/E ratio?\n"
            "- Market capitalization?\n"
            "- ROE and ROA?\n"
            "- Dividend yield?\n\n"

            "Do NOT use this tool for:\n"
            "- Revenue or net income values\n"
            "- Stock price history\n"
            "- Company performance explanation\n"
        )
    ),

    FunctionTool.from_defaults(
        fn=financial_statement_tool,
        name="financial_statement_tool",
        description=(
            "Use this tool ONLY when the user asks for exact structured financial numbers "
            "from income statement, balance sheet, or cash flow statement.\n\n"
            
            "INPUT SHOULD BE IN THE FOLLOWING STRUCTURED FORMAT:\n"
            " Date: YYYY-MM-DD\nCompany: Company Name\nFinancial Year: YYYY-YY\nQuarter: Q1/Q2/Q3/Q4/None\nQuestion: User Question\n\n"
            
            "Supports both quarterly data (Q1, Q2, Q3, Q4) and annual data.\n\n"

            "Examples:\n"
            "- What was revenue in Q2 2025-26?\n"
            "- Give net income for FY 2025-26.\n"
            "- Show EPS for Q3 2025-26.\n"
            "- Total assets in Q4 2025-26.\n\n"

            "This tool returns raw financial statement data.\n\n"

            "Do NOT use this tool for:\n"
            "- Explanations\n"
            "- Performance discussion\n"
            "- Management commentary\n"
            "- Stock price movement\n"
        )
    ),
]