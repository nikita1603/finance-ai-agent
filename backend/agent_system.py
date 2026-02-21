import os
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.google_genai import GoogleGenAI  # Official integration
from  ddgs import DDGS  # DuckDuckGo Search for real-time web info
from backend.rag_model import rag_tool
from backend.tools import  get_gnews_articles,financial_statement_tool, fundamental_tool, historical_price_tool 
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

import logging
logger = logging.getLogger(__name__)

# ---------------- Gemini LLM ----------------
# Use the official wrapper so the agent can handle tool-calling natively
llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
    api_key=os.getenv("AIzaSyCwEHySMJ26uyjQPiTUt0bFujXPpuzIz1U") # Recommended: set this in your environment
)


# ---------------- Tools ----------------
tools = [

    FunctionTool.from_defaults(
        fn=rag_tool,
        name="rag_tool",
       description=(
            "Use this tool ONLY for qualitative, narrative, or explanatory analysis "
            "from company documents such as earnings calls, investor presentations, "
            "annual reports, and quarterly reports.\n\n"

            "Examples:\n"
            "- Explain Q3 performance.\n"
            "- What were the key highlights this quarter?\n"
            "- Summarize management commentary.\n"
            "- Why did margins decline?\n"
            "- Headline from Q3 earnings call.\n\n"

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
            "Use this tool for sentiment and catalyst analysis.\n\n"
            "Do NOT paste full article content. Instead:\n"
            "- Extract headline\n"
            "- Provide a concise 1–2 line summary\n"
            "- Include the article hyperlink\n\n"
            "Only include news relevant to the company or broader market context."
        )
    ),

    FunctionTool.from_defaults(
        fn=historical_price_tool,
        name="historical_price_tool",
        description=(
            "Use this tool ONLY for historical stock price data for a specific date.\n\n"

            "Examples:\n"
            "- What was the stock price on 2026-01-15?\n"
            "- Price movement on a specific trading day.\n\n"

            "This tool retrieves OHLCV data for ONE specific date.\n\n"

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
        "Use this tool ONLY for company valuation and fundamental ratios.\n\n"

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

            "Examples:\n"
            "- What was revenue in Q2?\n"
            "- Give net income for FY 2025-26.\n"
            "- Show EPS for Q3.\n"
            "- EBITDA in FY2024.\n\n"

            "This tool returns raw financial statement data.\n\n"

            "Do NOT use this tool for:\n"
            "- Explanations\n"
            "- Performance discussion\n"
            "- Management commentary\n"
            "- Stock price movement\n"
        )
    ),
]


system_prompt = (
    "You are a Professional Equity Research Analyst AI.\n\n" 

    " You receive structured input in the following format:\n\n" 
    " Date: YYYY-MM-DD\n" 
    " Company: Company Name\n" 
    " Financial Year: YYYY-YY\n" 
    " Quarter: Q1/Q2/Q3/Q4/None\n" 
    " Question: User Question\n\n" 
    

    " CRITICAL RULES\n" 

    " 1. ALWAYS use tools for data retrieval.\n" 
    " 2. NEVER fabricate financial numbers.\n" 
    " 3. NEVER guess ratios, stock prices, revenue, or financial metrics.\n" 
    " 4. If data is unavailable, explicitly say: \"Data not available from provided sources.\"\n" 
    " 5. Use the MINIMUM number of tools required.\n" 
    " 6. Do NOT mix qualitative explanation with raw number tools unless necessary.\n" 
    " 7. If unsure which tool to use, analyze user intent carefully.\n" 
    " 8. If multiple data types are required, call tools sequentially.\n" 
    " 9. Never answer from memory.\n\n" 

    " Tool Selection Rules:\n" 

    " - Exact financial numbers → financial_statement_tool\n" 
    " - Valuation ratios or company fundamentals → fundamental_tool\n" 
    " - Historical stock price on specific date → historical_price_tool\n" 
    " - News, catalysts, or stock movement reasons → get_gnews_articles\n" 
    " - Management commentary, earnings discussion, qualitative analysis → rag_tool\n\n" 

    " When calling tools:\n" 

    " ALWAYS pass the FULL structured input exactly as received.\n\n" 
    " PLease answer 'Data not available from provided sources.' after exhausting all tools if you cannot find the answer.\n\n"
    
    " FINAL RESPONSE FORMAT\n" 

    " 1. Data Summary\n" 
    "    - Provide structured factual output from tools.\n" 
    "    - Use bullet points where helpful.\n\n" 
    " 2. Analytical Interpretation\n" 
    "    - Explain what the numbers or events imply.\n" 
    "    - Connect performance, valuation, and catalysts logically.\n" 
    "    - Be concise but analytical.\n\n" 
    " 3. Conclusion (2–3 lines)\n" 
    "    - Clear takeaway.\n" 
    "    - Neutral professional tone.\n" 
    "    - No speculation without evidence.\n"
)


# ---------------- FunctionAgent ----------------
agent = FunctionAgent(
    tools=tools,
    llm=llm, 
    system_prompt=system_prompt,
    verbose=True,
    max_function_calls=5
)


# ---------------- CLI ---------------
# async def main():
#     print("Finance AI Agent ready. Type 'exit' to quit.\n")
#     while True:
#         user_input = "Company: Hdfc\nQuestion: what was the revenue on 2025?"
#         if user_input.lower() == "exit":
#             break
#         try:
#             # Use chat() for conversation or chat_history maintenance
#             response = await agent.run(user_input) 
#             print("\nAnswer:\n", response, "\n")
#         except Exception as e:
#             print("Error:", e)

# if __name__ == "__main__":
#     print("Running agent...")
#     asyncio.run(main())
