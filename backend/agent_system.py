import os
from llama_index.core.agent import FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI  # Official integration
from backend.tools.company_financial_statement_tool.rag_model import rag_tool
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings
from dotenv import load_dotenv
import logging
from backend.tools.tools import TOOLS

load_dotenv()

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

logger = logging.getLogger(__name__)

# ---------------- Gemini LLM ----------------
# Use the official wrapper so the agent can handle tool-calling natively
llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

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

    " ALWAYS pass the FULL structured input exactly as received.\n" 
    " Please answer 'Data not available from provided sources.' after exhausting ALL tools if you cannot find the answer.\n\n"
    
    " FINAL RESPONSE FORMAT\n" 

    " 1. Data Summary\n" 
    "    - Provide structured factual output from tools.\n" 
    "    - Use bullet points where helpful.\n\n" 
    " 2. Analytical Interpretation\n" 
    "    - Explain what the numbers or events imply.\n" 
    "    - Connect performance, valuation, and catalysts logically.\n" 
    "    - Be concise but analytical.\n\n" 
    " 3. Conclusion (2-3 lines)\n" 
    "    - Clear takeaway.\n" 
    "    - Neutral professional tone.\n" 
    "    - No speculation without evidence.\n"
)


# ---------------- FunctionAgent ----------------
agent = FunctionAgent(
    tools=TOOLS,
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
