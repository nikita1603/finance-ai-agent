import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import Settings

import logging
logger = logging.getLogger(__name__)



# client = genai.Client(api_key=os.getenv("AIzaSyCwEHySMJ26uyjQPiTUt0bFujXPpuzIz1U"))  # or put your key directly
# gemini_model = "gemini-2.5-flash"

# ---------------- Local embeddings ----------------
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = None  # disable OpenAI

Settings.embed_model = embed_model

Settings.node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

def build_and_save(company_name):
    
    logger.info(f"Building index for company: {company_name}")
    logger.info(f"Reading documents from: C:/Users/Nikita/OneDrive/Desktop/Finance_project_folder/data/{company_name}")

    documents = SimpleDirectoryReader(
        input_dir=f"C:/Users/Nikita/OneDrive/Desktop/Finance_project_folder/data/{company_name}",
        recursive=True,
        filename_as_id=True
    ).load_data()

    logger.info(f"Read {len(documents)} documents for {company_name}. Building index...")
    index = VectorStoreIndex.from_documents(documents)

    logger.info(f"Index built successfully for {company_name}. Persisting to disk...")

    index.storage_context.persist(
        persist_dir=f"C:/Users/Nikita/OneDrive/Desktop/Finance_project_folder/vector_store/{company_name}"
    )

    logger.info(f"Index for {company_name} persisted successfully.")

if __name__ == "__main__":

    logger.info("Starting index building process for all companies...")

    build_and_save("Hdfc")
    build_and_save("Reliance")