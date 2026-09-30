"""Step 4 – Build and persist the FAISS vector store."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import VECTOR_STORE_DIR


def build_vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    persist: bool = True,
) -> FAISS:
    """Create a FAISS vector store from chunks and embeddings."""
    print(f"[VectorStore] Indexing {len(chunks)} chunk(s) into FAISS...")
    vector_db = FAISS.from_documents(chunks, embeddings)

    if persist:
        vector_db.save_local(str(VECTOR_STORE_DIR))
        print(f"[VectorStore] Saved index to {VECTOR_STORE_DIR}")

    return vector_db


def load_vector_store(
    embeddings: Embeddings,
    path: Path = VECTOR_STORE_DIR,
) -> FAISS:
    """Load a persisted FAISS index from disk."""
    print(f"[VectorStore] Loading index from {path}")
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )