# Finance AI Agent

An AI-powered agent for financial analysis and insights using Retrieval-Augmented Generation (RAG). The project combines a FastAPI backend with a Streamlit frontend to provide intelligent financial data retrieval and analysis for Indian equities.

## Features

- **RAG-based Financial Analysis**: LlamaIndex with Google Gemini (`gemini-2.5-flash`) for document retrieval and qualitative analysis from earnings calls, annual reports, and quarterly results
- **Quantitative Risk Analytics**: Sharpe ratio, Sortino ratio, CAGR, max drawdown, rolling volatility, and OLS beta vs NIFTY 50
- **Multi-company Support**: HDFC Bank and Reliance Industries, with pre-built vector stores per company
- **Five Specialized Tools**: Each tool is purpose-scoped to minimise hallucination and unnecessary LLM calls
- **REST API Backend**: FastAPI backend with auto-generated docs
- **Interactive Web UI**: Two-tab Streamlit interface — Chat with conversation history, and Visualize with interactive Plotly charts
- **Real-time Market Data**: yfinance integration for stock prices, fundamentals, and quantitative analytics
- **News Integration**: GNews API for event-driven and sentiment analysis
- **Evaluation Suite**: End-to-end agent evaluation with 95% confidence intervals and RAG hallucination detection

## Project Structure

```
.
├── backend/
│   ├── agent_system.py       # FunctionAgent with Gemini LLM and system prompt
│   ├── build_index.py        # Builds vector stores from PDFs in data/
│   ├── main.py               # FastAPI app
│   ├── logger_config.py
│   └── tools/
│       ├── tools.py                          # Tool registry (TOOLS list — 5 tools)
│       ├── company_financial_statement_tool/ # RAG tool (hybrid retrieval + reranking)
│       ├── company_fundamental_tool/         # Valuation ratios + dual beta via yfinance
│       ├── historical_price_tool/            # OHLCV data via yfinance
│       ├── news_tool/                        # News via GNews API
│       └── quant_analytics_tool/             # Risk/return analytics via yfinance + OLS
├── frontend/
│   └── app.py                # Streamlit UI (Chat + Visualize tabs)
├── evaluation/
│   ├── evaluate.py           # End-to-end agent evaluation (tool accuracy, CIs, latency)
│   ├── hallucination_eval.py # RAG hallucination detection using Gemini as judge
│   ├── eval_utils.py         # Result printing with 95% confidence intervals
│   ├── evaluation_sample.csv # Test cases for full agent evaluation
│   ├── rag_evaluation_sample.csv # Test cases for RAG-specific evaluations
│   └── makefile              # Evaluation commands
├── data/
│   ├── hdfc/                 # HDFC earnings PDFs, press releases, key parameters
│   └── reliance/             # Reliance earnings PDFs and press releases
├── vector_store/
│   ├── hdfc/                 # Pre-built LlamaIndex vector store for HDFC
│   └── reliance/             # Pre-built LlamaIndex vector store for Reliance
├── pyproject.toml
├── makefile
└── .env
```

## Agent Tools

The agent selects tools based on query intent. All tools share the same structured input format:

```
Date: YYYY-MM-DD
Company: Company Name
Financial Year: YYYY-YY
Quarter: Q1/Q2/Q3/Q4/None
Question: User Question
```

| Tool | Purpose |
|---|---|
| `rag_tool` | Financial figures and qualitative analysis from indexed earnings PDFs (revenue, PAT, NIM, GNPA, management commentary, guidance) |
| `fundamental_tool` | Current valuation ratios: P/E, P/B, ROE, market cap, dividend yield; published 5Y monthly beta and computed 1Y daily OLS beta vs NIFTY 50 |
| `historical_price_tool` | OHLCV stock price data for a specific date |
| `get_gnews_articles` | Recent news, stock movement reasons, event-driven analysis |
| `quant_analytics_tool` | Risk/return analytics over a period: CAGR, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, rolling 30d volatility, OLS beta/alpha vs NIFTY 50 |

The agent is capped at 10 tool calls per query to prevent runaway chains.

## Prerequisites

- Python 3.13+
- `pip` and `uv` package manager

## Setup

1. **Clone the repository**

2. **Create and activate the environment**
   ```bash
   make develop
   ```
   This will:
   - Create a Python 3.13 virtual environment
   - Upgrade pip and install uv
   - Install the project and all dependencies in editable mode

## Environment Variables

Create a `.env` file in the root directory:

```env
# Google Generative AI API Key (for Gemini models)
GEMINI_API_KEY=your_gemini_api_key_here

# GNews API Key (for news data retrieval)
GNEWS_API_KEY=your_gnews_api_key_here
```

- **GEMINI_API_KEY**: [Google AI Studio](https://aistudio.google.com/app/api-keys)
- **GNEWS_API_KEY**: [GNews API](https://gnews.io/)

## Running the Project

### Build Vector Index (first-time setup)
```bash
make build_index
```
Indexes all PDFs in `data/` and writes vector stores to `vector_store/`.

### Run Backend Server
```bash
make run_server
```
Starts the FastAPI server at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### Run Frontend UI
```bash
make run_ui
```
Starts the Streamlit web interface with Chat and Visualize tabs.

## Evaluation

The `evaluation/` module provides two complementary evaluation modes.

### 1. End-to-end Agent Evaluation

Runs the agent against a CSV of test cases and measures:

- **Tool accuracy** — whether the agent called exactly the expected tools
- **Tool precision / recall** — partial credit for correct tool selection
- **Keyword recall** — whether the response contains expected terms
- **Multi-hop accuracy** — whether the agent correctly chains multiple tools
- **Source precision / recall** — for RAG queries, whether the right documents were retrieved
- **Response rate** — fraction of queries that returned a non-empty answer
- **Latency** — avg, p95 (with bootstrap CI), and max query time in seconds

All aggregate metrics are reported with **95% confidence intervals** (Wilson score for binary metrics, percentile bootstrap for ratio averages and latency p95). A per-query timeout of 300 seconds prevents upstream API hangs from blocking the evaluation run.

```bash
make evaluate
```

### 2. RAG Hallucination Evaluation

Bypasses the agent and directly evaluates the RAG pipeline. Uses Gemini to detect claims in the generated answer that are not supported by the retrieved context.

```bash
make hallucination
```

Output includes a per-query hallucination flag, the specific hallucinated claims, and an overall hallucination rate.

### Latest Results

**1. End-to-end Agent Evaluation** (35 test cases, evaluated 2026-08-09)

| Metric | Score | 95% CI |
|---|---|---|
| Tool accuracy | 80% | [64%, 90%] |
| Tool precision | 91% | [85%, 97%] |
| Tool recall | 100% | [100%, 100%] |
| Keyword recall | 94% | [90%, 97%] |
| Multi-hop accuracy | 91% | [77%, 97%] |
| Response rate | 100% | [100%, 100%] |
| Latency avg | — | — |
| Latency p95 | 114.1s | — |

**2. RAG Hallucination Evaluation** (16 RAG test cases, evaluated 2026-08-09)

| Metric | Score |
|---|---|
| Cases evaluated | 16 |
| Hallucinations detected | 2 |
| Hallucination rate | 12.5% |

Two hallucinations were detected: one period-attribution error (figures cited for the wrong quarter) and one formatting defect where a number was presented without its correct unit. Neither involved fabricated data — both arose from context bleed in retrieval. The strict context-only prompt remains effective for fabrication prevention; retrieval precision is the active area for improvement.

Key observations:
- Tool recall improved to 100% — the agent never fails to call a required tool
- Tool precision at 91% reflects some over-calling on open-ended queries
- Keyword recall improved from 86% to 94% after adding explicit unit-preservation rules to the system prompt
- Latency p95 increased from 84.7s to 114.1s due to OLS regression computations in `quant_analytics_tool`

### Test Case CSV Format

Both `evaluation_sample.csv` and `rag_evaluation_sample.csv` share the same schema:

| Column | Description |
|---|---|
| `company` | Company name (e.g., `hdfc`, `reliance`) |
| `date` | Query date (`YYYY-MM-DD`) |
| `financial_year` | e.g., `2025-26` |
| `quarter` | `Q1`–`Q4` or `None` |
| `query` | Natural language question |
| `expected_tools_called` | Semicolon-separated expected tool names |
| `expected_sources_used` | Semicolon-separated expected PDF filenames (RAG cases only) |
| `is_multi_hop` | `true` if the query requires more than one tool |
| `expected_keywords` | Semicolon-separated keywords expected in the response |

## Notes

- The vector stores are pre-built; run `make build_index` only when adding new documents
- Never commit `.env` to version control
- To add a new company, add its PDFs under `data/<company>/`, add its ticker to `backend/tools/utils.py`, and re-run `make build_index`
- `quant_analytics_tool` resolves the date window in this order:
  1. **Explicit years** in the question ("in 2025", "2025 and 2026") — overrides everything
  2. **Relative phrases** ("last 6 months", "2 years") — applied relative to the end of the selected Financial Year (March 31), not today
  3. **No period specified** — defaults to the full Financial Year window (1 year ending March 31 of the selected FY)

  Example: selecting FY 2023-24 and asking "Sharpe ratio over the last year" fetches April 2023 – March 2024, not the most recent 12 months
