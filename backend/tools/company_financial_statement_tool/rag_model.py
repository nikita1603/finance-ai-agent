import os
from typing import Optional, Tuple

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from google import genai
from llama_index.core import load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from dotenv import load_dotenv

import logging
import time

"""RAG helper for company financial documents.

This module implements a retrieval-augmented generation (RAG) helper that:
- parses structured user queries,
- loads a company's vector store, retrieves relevant document nodes,
- calls the Gemini model with a context-limited prompt to produce grounded answers.

Retrieval pipeline:
  1. Hybrid search — vector (all-MiniLM-L6-v2) + BM25 fused via reciprocal rank fusion.
  2. Cross-encoder reranking — bge-reranker-base trims to top-N most relevant nodes.
  3. Gemini generation — strictly context-grounded answer.
"""

logger = logging.getLogger(__name__)
load_dotenv()

# ---------------- Gemini client ----------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = "gemini-2.5-flash"

# ---------------- Embeddings ----------------
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = None
Settings.embed_model = embed_model

# ---------------- Reranker (module-level to avoid reloading per call) ----------------
# Cross-encoder reranker: scores every (query, node) pair directly — much more
# accurate than bi-encoder similarity for final candidate selection.
reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-base", top_n=4)


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

    Uses hybrid retrieval (vector + BM25) fused via reciprocal rank fusion,
    then re-ranks with a cross-encoder to select the most relevant nodes.

    Returns a tuple `(nodes, context_text)`.
    """
    storage_context = StorageContext.from_defaults(persist_dir=f"vector_store/{company}")
    index = load_index_from_storage(storage_context)

    # Enrich query with quarter/year hints to narrow retrieval
    quarter_name_map = {"Q1": "quarter 1", "Q2": "quarter 2", "Q3": "quarter 3", "Q4": "quarter 4"}
    enriched_query = question
    if quarter and quarter != "None":
        enriched_query += f" (Related to {quarter_name_map.get(quarter, quarter)} ({quarter}))"
    if fy:
        enriched_query += f" (For financial year {fy})"

    logger.info(f"Retrieval query: {enriched_query}")

    # --- 1. Hybrid retrieval: vector + BM25 ---
    # Cast a wide net (top_k=12) before reranking trims to top 4.
    vector_retriever = index.as_retriever(similarity_top_k=12)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=list(index.docstore.docs.values()),
        similarity_top_k=12,
    )
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=12,
        # num_queries=1 disables LLM query-generation; use the original query only.
        num_queries=1,
        mode="reciprocal_rerank",
    )
    nodes = fusion_retriever.retrieve(enriched_query)

    # --- 2. Cross-encoder reranking ---
    # The reranker scores each (question, node) pair directly and returns top_n=4.
    nodes = reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(question))

    # Build context string with source metadata for traceability
    context_parts = []
    for node in nodes:
        source = node.metadata.get("file_name", "Unknown")
        page = node.metadata.get("page", "N/A")
        context_parts.append(f"[Source: {source}, Page: {page}]\n{node.text}")

    context = "\n\n".join(context_parts)
    return nodes, context


def _generate_answer(context: str, question: str, max_retries: int = 4) -> str:
    """Generate an answer from Gemini using only the provided context."""
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

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=gemini_model, contents=prompt)
            return response.text
        except Exception as e:
            is_last = attempt == max_retries - 1
            if "503" not in str(e) or is_last:
                raise
            delay = 2 ** attempt
            logger.warning(f"Gemini 503 on attempt {attempt + 1}/{max_retries}, retrying in {delay}s: {e}")
            time.sleep(delay)


def _format_chunks(nodes: list) -> str:
    """Format retrieved chunks cleanly when LLM synthesis is unavailable."""
    lines = ["Retrieved context (LLM synthesis unavailable):\n"]
    for i, node in enumerate(nodes, start=1):
        source = node.metadata.get("file_name", "Unknown")
        page = node.metadata.get("page", "N/A")
        lines.append(f"[{i}] Source: {source}, Page: {page}\n{node.text}\n")
    return "\n".join(lines)


def rag_tool(user_query: str, eval_mode: bool = False) -> str:
    """Public RAG tool entrypoint used by the agent/tooling layer.

    Parses the structured user query, retrieves relevant context from the
    company's vector store using hybrid search + reranking, asks the Gemini
    model to answer using only that context, and returns the result.
    When `eval_mode` is True the raw model answer is returned.
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

        if not nodes:
            return "Not found in documents."

        logger.info("Top Retrieved Documents (after reranking):")
        for i, node in enumerate(nodes, start=1):
            logger.info(
                f"{i}. Source: {node.metadata.get('file_name', 'Unknown')}, "
                f"Page: {node.metadata.get('page', 'N/A')}\n"
                f"--- chunk ---\n{node.text}\n-------------"
            )

    except Exception as e:
        logger.error(f"Error during retrieval in rag_tool: {e}")
        return "Error retrieving information from documents."

    try:
        answer = _generate_answer(context, question)
    except Exception as e:
        logger.error(f"Error during generation in rag_tool: {e}")
        unique_sources = list(dict.fromkeys(
            node.metadata.get("file_name", "Unknown") for node in nodes
        ))
        sources_marker = "\n\n[SOURCES_USED: " + "; ".join(unique_sources) + "]"
        context_marker = "\n\n[RAG_CONTEXT_START]\n" + context + "\n[RAG_CONTEXT_END]"
        return _format_chunks(nodes) + sources_marker + context_marker

    if eval_mode:
        return answer

    unique_sources = list(dict.fromkeys(
        node.metadata.get("file_name", "Unknown") for node in nodes
    ))
    sources_marker = "\n\n[SOURCES_USED: " + "; ".join(unique_sources) + "]"
    context_marker = "\n\n[RAG_CONTEXT_START]\n" + context + "\n[RAG_CONTEXT_END]"
    return answer + sources_marker + context_marker
