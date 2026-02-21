import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from google import genai  # modern package
from llama_index.core import load_index_from_storage
from llama_index.core.storage import StorageContext

import logging
logger = logging.getLogger(__name__)


# ---------------- Gemini client ----------------
client = genai.Client(api_key=os.getenv("AIzaSyCwEHySMJ26uyjQPiTUt0bFujXPpuzIz1U"))  # or put your key directly
gemini_model = "gemini-2.5-flash"

# ---------------- Local embeddings ----------------
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = None  # disable OpenAI
Settings.embed_model = embed_model


def rag_tool(user_query: str) -> str:

    try:
        logger.info(f"Using RAG tool for query: {user_query}")

        # ---------- Parse structured input ----------
        parsed = {
            "date": None,
            "company": None,
            "financial_year": None,
            "quarter": None,
            "question": None
        }

        for line in user_query.split("\n"):
            if line.startswith("Date:"):
                parsed["date"] = line.replace("Date:", "").strip()
            elif line.startswith("Company:"):
                parsed["company"] = line.replace("Company:", "").strip()
            elif line.startswith("Financial Year:"):
                parsed["financial_year"] = line.replace("Financial Year:", "").strip()
            elif line.startswith("Quarter:"):
                parsed["quarter"] = line.replace("Quarter:", "").strip()
            elif line.startswith("Question:"):
                parsed["question"] = line.replace("Question:", "").strip()

        company = parsed["company"]
        fy = parsed["financial_year"]
        quarter = parsed["quarter"]
        question = parsed["question"]

        logger.info(f"Parsed company: {company}, FY: {fy}, Quarter: {quarter}, Question: {question} ")

        # ---------- Load Company Index ----------
        storage_context = StorageContext.from_defaults(
            persist_dir=f"C:/Users/Nikita/OneDrive/Desktop/Finance_project_folder/vector_store/{company}"
        )

        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever(similarity_top_k=5)

        # ---------- Enrich retrieval query ----------
        enriched_query = question

        if quarter and quarter != "None":
            enriched_query += f" related to {quarter}"

        if fy:
            enriched_query += f" for financial year {fy}"

        logger.info(f"Retrieval query: {enriched_query}")

        nodes = retriever.retrieve(enriched_query)

        if not nodes:
            return "Not found in documents."


        logger.info("Top Retrieved Documents:")
        for i, node in enumerate(nodes, start=1):
            logger.info(f"{i}. Source: {node.metadata.get('file_name', 'Unknown')}, Page: {node.metadata.get('page', 'N/A')}")
            logger.info(f"   Relevance Score: {node.metadata.get('relevance_score', 'N/A')}")
            logger.info(f"   Content Snippet: {node.text[:200]}...")  # log first 200 chars

        # ---------- Build Context ----------
        context_parts = []

        for node in nodes:
            source = node.metadata.get("file_name", "Unknown")
            page = node.metadata.get("page", "N/A")

            context_parts.append(
                f"[Source: {source}, Page: {page}]\n{node.text}"
            )

        context = "\n\n".join(context_parts)

        # ---------- Final LLM Prompt ----------
        prompt = ("You are analyzing company documents.\n\n" 

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
        " Provide a clear and structured answer.\n" 
        " Cite sources in this format when relevant:\n" 
        " (Source: file_name, Page: page_number)\n"
        )

        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt
        )

        return response.text

    except Exception as e:
        logger.error(f"Error in rag_tool: {e}")
        return "Error retrieving information from documents."