# Finance AI Agent

An AI-powered agent for financial analysis and insights using Retrieval-Augmented Generation (RAG). The project combines a FastAPI backend with a Streamlit frontend to provide intelligent financial data retrieval and analysis for Indian equities.

## Features

- **RAG-based Financial Analysis**: Uses LlamaIndex with Google Gemini (`gemini-2.5-flash`) for document retrieval and qualitative analysis from earnings calls, annual reports, and quarterly results
- **Multi-company Support**: HDFC Bank and Reliance Industries, with pre-built vector stores per company
- **Five Specialized Tools**: Each tool is purpose-scoped to minimize hallucination and unnecessary LLM calls
- **REST API Backend**: FastAPI backend with auto-generated docs
- **Interactive Web UI**: Streamlit-based chat interface
- **Real-time Market Data**: yfinance integration for stock prices, financials, and fundamentals
- **News Integration**: GNews API for event-driven and sentiment analysis
- **Evaluation Suite**: End-to-end agent evaluation and RAG hallucination detection

## Project Structure

```
.
├── backend/
│   ├── agent_system.py       # FunctionAgent with Gemini LLM and system prompt
│   ├── build_index.py        # Builds vector stores from PDFs in data/
│   ├── main.py               # FastAPI app
│   ├── logger_config.py
│   └── tools/
│       ├── tools.py                          # Tool registry (TOOLS list)
│       ├── company_financial_statement_tool/ # RAG tool + yfinance financials
│       ├── company_fundamental_tool/         # Valuation ratios via yfinance
│       ├── historical_price_tool/            # OHLCV data via yfinance
│       └── news_tool/                        # News via GNews API
├── frontend/
│   └── app.py                # Streamlit UI
├── evaluation/
│   ├── evaluate.py           # End-to-end agent evaluation (tool accuracy, keywords, multi-hop)
│   ├── hallucination_eval.py # RAG hallucination detection using Gemini
│   ├── eval_utils.py         # Result printing utilities
│   ├── evaluation_sample.csv # Test cases for agent evaluation
│   ├── rag_evaluation_sample.csv # Test cases for RAG/hallucination evaluation
│   └── makefile              # Evaluation commands
├── data/
│   ├── hdfc/                 # HDFC earnings PDFs, annual report, press releases
│   └── reliance/             # Reliance earnings PDFs and annual report
├── vector_store/
│   ├── hdfc/                 # Pre-built LlamaIndex vector store for HDFC
│   └── reliance/             # Pre-built LlamaIndex vector store for Reliance
├── pyproject.toml
├── makefile
└── .env
```

## Agent Tools

The agent selects tools based on query intent. Each tool expects a structured input:

```
Date: YYYY-MM-DD
Company: Company Name
Financial Year: YYYY-YY
Quarter: Q1/Q2/Q3/Q4/None
Question: User Question
```

| Tool | Purpose |
|---|---|
| `rag_tool` | Qualitative analysis from earnings calls, annual reports, investor presentations |
| `financial_statement_tool` | Exact numbers from income statement, balance sheet, or cash flow (via yfinance) |
| `fundamental_tool` | Valuation ratios: P/E, P/B, ROE, market cap, dividend yield, beta |
| `historical_price_tool` | OHLCV stock price data for a specific date |
| `get_gnews_articles` | Recent news, stock movement reasons, event-driven analysis |

The agent is capped at 5 tool calls per query to prevent runaway chains.

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
Starts the Streamlit web interface.

### Run Both Simultaneously
```bash
# Terminal 1
make run_server

# Terminal 2
make run_ui
```

## Evaluation

The `evaluation/` module provides two complementary evaluation modes.

### End-to-end Agent Evaluation

Runs the agent against a CSV of test cases and measures:

- **Tool accuracy** — whether the agent called the expected tools
- **Tool precision / recall** — partial credit for correct tool selection
- **Keyword recall** — whether the response contains expected terms
- **Multi-hop accuracy** — whether the agent correctly chains multiple tools
- **Source precision / recall** — for RAG queries, whether the right documents were retrieved
- **Response rate** — fraction of queries that returned a non-empty answer

```bash
# From repo root
make evaluate

# Or with a custom CSV
cd evaluation && make evaluate CSV_PATH=my_test_cases.csv
```

### RAG Hallucination Evaluation

Bypasses the agent and directly evaluates the RAG pipeline (`_retrieve_context` → `_generate_answer`). Uses Gemini to detect claims in the generated answer that are not supported by the retrieved context.

```bash
cd evaluation && make hallucination
```

Output includes a per-query hallucination flag, the specific hallucinated claims, and a summary rate.

## Evaluation Results

### End-to-end Agent Evaluation

| Metric | Score |
|---|---|
| Total tests | 30 |
| Tools accuracy | 67% |
| Tools precision | 77% |
| Tools recall | 75% |
| Keywords recall | 46% |
| Multi-hop accuracy | 90% |
| Response rate | 100% |

### RAG Hallucination Evaluation

| Metric | Score |
|---|---|
| RAG test cases evaluated | 12 |
| Successfully checked | 4 |
| Hallucinations detected | 1 |
| Hallucination rate | 25.0% |

### Test Case CSV Format

`evaluation_sample.csv` columns:

| Column | Description |
|---|---|
| `company` | Company name (e.g., `HDFC`, `Reliance`) |
| `date` | Query date (`YYYY-MM-DD`) |
| `financial_year` | e.g., `2025-26` |
| `quarter` | `Q1`–`Q4` or empty |
| `query` | Natural language question |
| `expected_tools_called` | Semicolon-separated expected tool names |
| `expected_sources_used` | Semicolon-separated expected PDF source names (RAG only) |
| `is_multi_hop` | `true`/`false` |
| `expected_keywords` | Semicolon-separated keywords expected in the response |

## Notes

- The vector stores are pre-built; run `make build_index` only when adding new documents
- Never commit `.env` to version control
- To add a new company, add its PDFs under `data/<company>/` and re-run `make build_index`
