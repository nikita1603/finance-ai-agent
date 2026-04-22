"""Build and persist vector search indexes from local documents.

This script reads documents from a company's folder under `data/`, builds
a vector index using local HuggingFace embeddings, and persists the
resulting vector store to `vector_store/{company}`. The file intentionally
uses `Settings` to configure the llama_index runtime to avoid external LLM
calls and to use a local embedding model.
"""

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import os


import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------- Local embeddings configuration ----------------
# Use a local sentence transformer for embedding generation.
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Prevent the library from trying to call an external LLM provider
Settings.llm = None  # disable OpenAI

# Register the local embedder with the library settings
Settings.embed_model = embed_model

# Configure node parsing (how documents are chunked for indexing)
Settings.node_parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)


def build_and_save(company_name):
    """Build a VectorStoreIndex for `company_name` and persist it.

    - Reads all files under `data/{company_name}`
    - Builds a vector index using the configured embedding model
    - Persists the index to `vector_store/{company_name}`

    Args:
        company_name: directory name under `data/` containing documents.
    """
    logger.info(f"Building index for company: {company_name}")
    logger.info(f"Reading documents from: data/{company_name}")

    # Load documents from the company's data directory
    documents = SimpleDirectoryReader(
        input_dir=f"data/{company_name}",
        recursive=True,
        filename_as_id=True
    ).load_data()

    logger.info(f"Read {len(documents)} documents for {company_name}. Building index...")

    # Build the vector index from loaded documents
    index = VectorStoreIndex.from_documents(documents)

    logger.info(f"Index built successfully for {company_name}. Persisting to disk...")

    # Ensure the output directory exists and persist the storage context
    if not os.path.exists(f"vector_store/{company_name}"):
        os.makedirs(f"vector_store/{company_name}")
    index.storage_context.persist(
        persist_dir=f"vector_store/{company_name}"
    )

    logger.info(f"Index for {company_name} persisted successfully.")


if __name__ == "__main__":

    print("Starting index building process for all companies...")
    logger.info("Starting index building process for all companies...")

    # Example usage: build indexes for two companies included in the repo
    build_and_save("hdfc")
    build_and_save("reliance")