import os
from typing import List, Optional, Tuple

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from google import genai  # modern package
from llama_index.core import load_index_from_storage
from llama_index.core.storage import StorageContext
from dotenv import load_dotenv

import logging
logger = logging.getLogger(__name__)
load_dotenv()

# ---------------- Gemini client ----------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = "gemini-2.5-flash"

# ---------------- Local embeddings ----------------
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = None  # disable OpenAI
Settings.embed_model = embed_model


def _parse_query(user_query: str) -> dict:
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
    """Load vector store for company and retrieve relevant nodes + built context string."""
    storage_context = StorageContext.from_defaults(persist_dir=f"vector_store/{company}")
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=5)

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

    context_parts = []
    for node in nodes:
        source = node.metadata.get("file_name", "Unknown")
        page = node.metadata.get("page", "N/A")
        context_parts.append(f"[Source: {source}, Page: {page}]\n{node.text}")

    context = "\n\n".join(context_parts)
    return nodes, context


def _generate_answer(context: str, question: str) -> str:
    """Call Gemini to answer question grounded in retrieved context."""
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
    response = client.models.generate_content(model=gemini_model, contents=prompt)
    return response.text


def rag_tool(user_query: str, eval_mode: bool = False) -> str:
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

        logger.info("Top Retrieved Documents:")
        for i, node in enumerate(nodes, start=1):
            logger.info(f"{i}. Source: {node.metadata.get('file_name', 'Unknown')}, Page: {node.metadata.get('page', 'N/A')}")

        answer = _generate_answer(context, question)

        if eval_mode:
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
