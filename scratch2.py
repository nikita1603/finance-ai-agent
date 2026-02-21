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


# ---------------- RAG function ----------------
def rag_tool(user_query: str) -> str:

    logger.info(f"Using RAG tool for query: {user_query}")

    lines = user_query.split("\n")
    date = lines[0].replace("Date:", "").strip()
    company = lines[1].replace("Company:", "").strip()
    question = lines[2].replace("Question:", "").strip()

    logger.info(f"Parsed company: {company}, question: {question}")

    logger.info(f"Loading index for company: {company}")
    
    storage_context = StorageContext.from_defaults(
    persist_dir=f"C:/Users/Nikita/OneDrive/Desktop/Finance_project_folder/vector_store/{company}")

    index = load_index_from_storage(storage_context)

    logger.info(f"Index loaded successfully for {company}")

    retriever = index.as_retriever(similarity_top_k=10)

    logger.info(f"Retrieving relevant documents for the question...")

    nodes = retriever.retrieve(question)

    logger.info(f"Retrieved {len(nodes)} relevant nodes from the index")

    context_parts = []
    for node in nodes:
        source = node.metadata.get("file_name", "Unknown")
        page = node.metadata.get("page", "N/A")
        context_parts.append(f"[Source: {source}]\n[Page: {page}]\n{node.text}")

    logger.info(f"Constructed context from retrieved nodes")    

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer using ONLY the context and source info.
If the user asks about the title or document name, use the Source field.
If not found, say 'Not found in documents.'

Context:
{context}

Question: {question}
"""
    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt
    )

    logger.info(f"Received response from Gemini model")

    return response.text


# ---------------- CLI loop ----------------
if __name__ == "__main__":
    print("Type your question, or 'exit' to quit")
    while True:
        q = input("\nAsk: ")
        if q.lower() == "exit":
            break
        print("\nAnswer:", rag_tool(q))
