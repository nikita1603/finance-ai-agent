# Finance AI Agent

An AI-powered agent for financial analysis and insights using Retrieval-Augmented Generation (RAG). This project combines FastAPI backend with Streamlit frontend to provide intelligent financial data retrieval and analysis capabilities.

## Features

- **RAG-based Financial Analysis**: Uses LLaMA Index with Google Generative AI for intelligent document retrieval and analysis
- **Multi-company Support**: Support for analyzing financial data from multiple companies (e.g., Hdfc, Reliance)
- **Vector Store Integration**: Leverages FAISS-based vector stores for efficient document indexing and retrieval
- **REST API Backend**: FastAPI-based REST API for backend services
- **Interactive Web UI**: Streamlit-based web interface for user interactions
- **Real-time Data**: Integration with yfinance for real-time financial data

## Project Structure

```
.
├── backend/           # FastAPI backend application
├── frontend/          # Streamlit frontend application
├── data/              # Company financial documents
├── vector_store/      # Pre-built vector stores for different companies
├── pyproject.toml     # Project configuration
├── makefile          # Build and run commands
└── .env              # Environment variables
```

## Prerequisites

- Python 3.13+
- pip and uv package manager

## Setup

1. **Clone the repository**

2. **Create and activate environment**
   ```bash
   make develop
   ```
   This command will:
   - Create a Python 3.13 virtual environment
   - Upgrade pip and install uv
   - Install the project and its dependencies in editable mode

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Google Generative AI API Key (for Gemini models)
GEMINI_API_KEY=your_gemini_api_key_here

# GNews API Key (for news data retrieval)
GNEWS_API_KEY=your_gnews_api_key_here
```

### Getting API Keys

- **GEMINI_API_KEY**: Get it from [Google AI Studio](https://aistudio.google.com/app/api-keys)
- **GNEWS_API_KEY**: Get it from [GNews API](https://gnews.io/)

## Running the Project

### Run Backend Server
```bash
make run_server
```
This starts the FastAPI backend server on `http://localhost:8000`
- API documentation available at `http://localhost:8000/docs`

### Run Frontend UI
```bash
make run_ui
```
This starts the Streamlit web interface

### Run Both Simultaneously
You can run both commands in separate terminals:
- Terminal 1: `make run_server`
- Terminal 2: `make run_ui`

## Notes

- Ensure your `.env` file is never committed to version control
- The vector stores are pre-built for efficient retrieval
- Modify the backend tools in `backend/tools.py` to customize agent behavior
