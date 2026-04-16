import os
from typing import List, Optional, Tuple

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from google import genai  # modern package
from llama_index.core import load_index_from_storage
from llama_index.core.storage import StorageContext
from dotenv import load_dotenv

import logging

"""RAG helper for company financial documents.

This module implements a retrieval-augmented generation (RAG) helper that:
- parses structured user queries,
- loads a company's vector store, retrieves relevant document nodes,
- calls the Gemini model with a context-limited prompt to produce grounded answers.

The functions here are intentionally small and synchronous where possible;
the actual model call is performed via the Gemini client.
"""

logger = logging.getLogger(__name__)
load_dotenv()

# ---------------- Gemini client ----------------
# Client for calling Google's Gemini model via the `genai` package.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = "gemini-2.5-flash"

# ---------------- Local embeddings ----------------
# Use a local HuggingFace sentence transformer for embeddings and disable
# remote LLMs in the llama_index Settings to avoid external calls during
# embedding/index construction.
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = None  # disable OpenAI
Settings.embed_model = embed_model


def _parse_query(user_query: str) -> dict:
    """Parse a structured user query into components.

    Expected input lines:
        Date: YYYY-MM-DD
        Company: Company Name
        Financial Year: YYYY-YY
        Quarter: Q1/Q2/Q3/Q4/None
        Question: User Question

    Returns a dict with keys: date, company, financial_year, quarter, question.
    Company names are lower-cased to match storage folder names.
    """
    parsed = {
        "date": None,
        "company": None,
        "financial_year": None,
        "quarter": None,
        "question": None
    }

    # Split on newlines and extract known prefixed fields
    for line in user_query.split("\n"):
        line = line.strip()
        if line.startswith("Date:"):
            parsed["date"] = line.replace("Date:", "").strip()
        elif line.startswith("Company:"):
            parsed["company"] = line.replace("Company:", "").strip().lower()
        elif line.startswith("Financial Year:"):
            parsed["financial_year"] = line.replace("Financial Year:", "").strip()
        elif line.startswith("Quarter:"):
            parsed["quarter"] = line.replace("Quarter:", "").strip()
        elif line.startswith("Question:"):
            parsed["question"] = line.replace("Question:", "").strip()

    return parsed


def _retrieve_context(
    company: str,
    fy: Optional[str],
    quarter: Optional[str],
    question: str,
) -> Tuple[list, str]:
    """Load the company's vector store and retrieve top matching nodes.

    The function builds an `enriched_query` by appending quarter/year hints
    to improve retrieval relevance, fetches top-k nodes, and assembles a
    plain-text context string that will be provided to the LLM.

    Returns a tuple `(nodes, context_text)`.
    """
    storage_context = StorageContext.from_defaults(persist_dir=f"vector_store/{company}")
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=5)

    # Enrich the raw question with quarter/year to narrow retrieval
    enriched_query = question
    if quarter and quarter != "None":
        quarter_name_map = {
            "Q1": "quarter 1",
            "Q2": "quarter 2",
            "Q3": "quarter 3",
            "Q4": "quarter 4"
        }
        enriched_query += f" (Related to {quarter_name_map.get(quarter, quarter)} ({quarter}))"
    if fy:
        enriched_query += f" (For financial year {fy})"

    logger.info(f"Retrieval query: {enriched_query}")
    nodes = retriever.retrieve(enriched_query)

    # Build a context string that includes source metadata for traceability
    context_parts = []
    for node in nodes:
        source = node.metadata.get("file_name", "Unknown")
        page = node.metadata.get("page", "N/A")
        context_parts.append(f"[Source: {source}, Page: {page}]\n{node.text}")

    context = "\n\n".join(context_parts)
    return nodes, context


def _generate_answer(context: str, question: str) -> str:
    """Generate an answer from Gemini using only the provided context.

    The prompt enforces strict instructions to avoid hallucination and
    to request exact failure messages when information is missing.
    """
    prompt = (
        "You are analyzing company documents.\n\n"
        " STRICT INSTRUCTIONS:\n"
        " - Use ONLY the provided context.\n"
        " - Do NOT use outside knowledge.\n"
        " - Do NOT infer numerical values unless explicitly mentioned.\n"
        " - If the answer is not explicitly stated, respond exactly: \"Not found in documents.\"\n"
        " - If numerical data is required but not found, respond exactly: \"Exact numerical data not found in documents.\"\n\n"
        " CONTEXT\n"
        f"{context}\n\n"
        " QUESTION\n"
        f"{question}\n\n"
        " INSTRUCTIONS\n"
        " - Provide a clear and structured answer.\n"
        " - Cite sources in this format when relevant: (Source: file_name, Page: page_number)\n"
    )

    # Call the Gemini model with the constructed prompt and return text
    response = client.models.generate_content(model=gemini_model, contents=prompt)
    return response.text


def rag_tool(user_query: str, eval_mode: bool = False) -> str:
    """Public RAG tool entrypoint used by the agent/tooling layer.

    Parses the structured user query, retrieves relevant context from the
    company's vector store, asks the Gemini model to answer using only
    that context, and returns the result. When `eval_mode` is True the raw
    model answer is returned (useful for automated evaluation).
    """
    logger.info(f"Using RAG tool for query: {user_query}")

    try:
        parsed = _parse_query(user_query)
        company = parsed["company"]
        fy = parsed["financial_year"]
        quarter = parsed["quarter"]
        question = parsed["question"]

        logger.info(f"Parsed company: {company}, FY: {fy}, Quarter: {quarter}, Question: {question}")

        nodes, context = _retrieve_context(company, fy, quarter, question)

        # If no nodes retrieved, return a specific 'not found' string
        if not nodes:
            return "Not found in documents."

        logger.info("Top Retrieved Documents:")
        for i, node in enumerate(nodes, start=1):
            logger.info(f"{i}. Source: {node.metadata.get('file_name', 'Unknown')}, Page: {node.metadata.get('page', 'N/A')}")

        # Generate the grounded answer using the retrieved context
        answer = _generate_answer(context, question)

        if eval_mode:
            # In evaluation mode return the raw model output
            return answer

        unique_sources = list(dict.fromkeys(
            node.metadata.get("file_name", "Unknown") for node in nodes
        ))
        sources_marker = "\n\n[SOURCES_USED: " + "; ".join(unique_sources) + "]"
        context_marker = "\n\n[RAG_CONTEXT_START]\n" + context + "\n[RAG_CONTEXT_END]"
        return answer + sources_marker + context_marker

    except Exception as e:
        logger.error(f"Error in rag_tool: {e}")
        return "Error retrieving information from documents."
