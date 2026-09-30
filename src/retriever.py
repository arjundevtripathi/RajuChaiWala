"""Step 5 – Retrieve relevant chunks for a question."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import TOP_K


def retrieve(
    vector_db: FAISS,
    question: str,
    k: int = TOP_K,
) -> list[Document]:
    """Return the top-k most relevant chunks for the question."""
    results = vector_db.similarity_search(question, k=k)
    print(f"[Retriever] Retrieved {len(results)} chunk(s) for: {question!r}")
    return results


def build_context(results: list[Document]) -> str:
    """Join retrieved chunks into a single context string."""
    return "\n\n".join(r.page_content for r in results)