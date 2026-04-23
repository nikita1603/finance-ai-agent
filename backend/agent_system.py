"""Agent wiring for the finance RAG assistant.

This module configures the LLM, callback handlers, and the
`FunctionAgent` used by the HTTP API.
"""

import os
from llama_index.core.agent import FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI  # Official integration
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings
from dotenv import load_dotenv
import logging
from backend.tools.tools import TOOLS

# Load environment variables from a .env file if present
load_dotenv()

# Configure a debug callback handler to capture detailed traces during
# agent execution. These callbacks are helpful when diagnosing tool calls
# and LLM interactions.
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
# Register the callback manager so the llama_index runtime emits events
# to the provided handlers.
Settings.callback_manager = callback_manager

# Module logger
logger = logging.getLogger(__name__)

# ---------------- Gemini LLM ----------------
# Use the official Google GenAI wrapper so the agent can call functions/tools
# natively via the model's tool-calling interface.
llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
)

# System prompt that guides the agent's high-level behavior and rules.
system_prompt = (
    "You are a Professional Equity Research Analyst AI.\n\n"

    "You receive structured input in the following format:\n"
    "  Date: YYYY-MM-DD\n"
    "  Company: Company Name\n"
    "  Financial Year: YYYY-YY\n"
    "  Quarter: Q1/Q2/Q3/Q4/None\n"
    "  Question: User Question\n\n"

    "CRITICAL RULES\n"
    "1. ALWAYS use tools for data retrieval — never answer from memory.\n"
    "2. NEVER fabricate or estimate financial numbers, ratios, or stock prices.\n"
    "3. Pass the FULL structured input block to every tool exactly as received.\n"
    "4. Use the MINIMUM number of tools required to answer the question.\n"
    "5. If multiple data types are needed, call tools sequentially.\n\n"

    "TOOL SELECTION GUIDE\n"
    "- Financial figures from filings (revenue, PAT, NIM, EPS, GNPA, capital adequacy, "
    "deposits, advances) OR management commentary, earnings discussion, strategic guidance "
    "→ rag_tool\n"
    "- Current valuation ratios (P/E, P/B, Market Cap, ROE, dividend yield) "
    "→ fundamental_tool\n"
    "- Historical stock price on a specific date → historical_price_tool\n"
    "- News, catalysts, regulatory updates, or stock movement reasons → get_gnews_articles\n\n"


    "WHEN DATA IS NOT FOUND\n"
    "Do NOT simply say 'data not available'. Instead, provide a helpful response that:\n"
    "  a) States which tools were called and what period/company was searched.\n"
    "  b) Explains the likely reason the data was not found — for example:\n"
    "     - 'Documents for this quarter may not be indexed in the vector store.'\n"
    "     - 'This metric is not typically disclosed in earnings presentations.'\n"
    "     - 'The requested date falls outside the available document range.'\n"
    "  c) Suggests what the user can do — for example:\n"
    "     - 'Try asking about a different quarter where data is available.'\n"
    "     - 'Check the annual report for FY2024-25 for this figure.'\n"
    "     - 'This valuation ratio is available via fundamental_tool for current data.'\n\n"

    "FINAL RESPONSE FORMAT\n"
    "1. Data Summary\n"
    "   - Structured factual output from tools, using bullet points where helpful.\n\n"
    "2. Analytical Interpretation\n"
    "   - What the numbers or events imply.\n"
    "   - Connect performance, valuation, and catalysts logically.\n"
    "   - Concise but analytical.\n\n"
    "3. Conclusion (2-3 lines)\n"
    "   - Clear takeaway in neutral professional tone.\n"
    "   - No speculation without evidence.\n"
)


# ---------------- FunctionAgent ----------------
# Create the FunctionAgent with the available tools and configured LLM.
# The agent will follow `system_prompt` and is allowed a limited number of
# function calls per query (controlled by `max_function_calls`).
agent = FunctionAgent(
    tools=TOOLS,
    llm=llm, 
    system_prompt=system_prompt,
    verbose=True,
    max_function_calls=10
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
