"""Build and persist vector search indexes from local documents.

This script reads documents from a company's folder under `data/`, builds
a vector index using local HuggingFace embeddings, and persists the
resulting vector store to `vector_store/{company}`. The file intentionally
uses `Settings` to configure the llama_index runtime to avoid external LLM
calls and to use a local embedding model.
"""

import json
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings


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
    chunk_overlap=100
)


def build_and_save(company_name):
    """Build a VectorStoreIndex for `company_name` and persist it.

    - Reads all files under `data/{company_name}`
    - On first run: builds index from scratch
    - On subsequent runs: embeds only new/changed files, removes deleted ones
    - Persists the index to `vector_store/{company_name}`

    Args:
        company_name: directory name under `data/` containing documents.
    """
    data_dir = f"data/{company_name}"
    persist_dir = f"vector_store/{company_name}"

    if not os.path.exists(data_dir):
        logger.warning(f"Data folder not found for '{company_name}': {data_dir} — skipping.")
        return

    logger.info(f"Building index for company: {company_name}")

    documents = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,
        filename_as_id=True
    ).load_data()

    logger.info(f"Read {len(documents)} document(s) for {company_name}.")

    if os.path.exists(persist_dir):
        # Load existing index
        logger.info("Existing index found. Checking for new/changed/deleted files...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

        # Embed new or changed files
        refreshed = index.refresh_ref_docs(documents)
        new_count = sum(refreshed)
        logger.info(f"{new_count} new/updated file(s) embedded. {len(refreshed) - new_count} skipped.")

        # Remove deleted files from the index
        current_doc_ids = {doc.doc_id for doc in documents}
        indexed_doc_ids = set(index.ref_doc_info.keys())
        deleted_ids = indexed_doc_ids - current_doc_ids
        for doc_id in deleted_ids:
            index.delete_ref_doc(doc_id, delete_from_docstore=True)
        if deleted_ids:
            logger.info(f"{len(deleted_ids)} deleted file(s) removed from index: {deleted_ids}")
    else:
        # First run — build from scratch
        logger.info("No existing index. Building from scratch...")
        os.makedirs(persist_dir)
        index = VectorStoreIndex.from_documents(documents)
        logger.info(f"Index built successfully for {company_name}.")

    index.storage_context.persist(persist_dir=persist_dir)
    logger.info(f"Index for {company_name} persisted successfully.")

    # Persist node corpus for BM25 so rag_model.py can rebuild BM25Retriever
    # from disk without re-embedding. Stored as JSON (text + metadata) rather
    # than pickling BM25Retriever directly, because pystemmer uses Cython
    # objects that cannot be serialized with standard pickle.
    nodes_data = [
        {"id_": node.node_id, "text": node.text, "metadata": node.metadata}
        for node in index.docstore.docs.values()
    ]
    bm25_nodes_path = os.path.join(persist_dir, "bm25_nodes.json")
    with open(bm25_nodes_path, "w", encoding="utf-8") as f:
        json.dump(nodes_data, f)
    logger.info(f"BM25 node corpus persisted to {bm25_nodes_path} ({len(nodes_data)} nodes).")


if __name__ == "__main__":
    logger.info("Starting index building process...")

    # Auto-discover all company folders under data/
    if not os.path.exists("data"):
        logger.error("'data/' directory not found. Exiting.")
    else:
        companies = [
            d for d in os.listdir("data")
            if os.path.isdir(os.path.join("data", d))
        ]
        if not companies:
            logger.warning("No company folders found under 'data/'.")
        else:
            logger.info(f"Found {len(companies)} company folder(s): {companies}")
            for company in companies:
                build_and_save(company)