"""Full RAG pipeline: retrieve → generate (Groq edition)."""

from langchain_community.vectorstores import FAISS

from src.embeddings import get_embeddings
from src.loader import load_document
from src.llm import build_prompt, get_llm
from src.retriever import build_context, retrieve
from src.splitter import split_documents
from src.vector_store import build_vector_store, load_vector_store


def build_index() -> FAISS:
    """Load → split → embed → store."""
    documents = load_document()
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    return build_vector_store(chunks, embeddings)


def get_index(reuse: bool = True) -> FAISS:
    """Load a saved index if available, otherwise build a fresh one."""
    embeddings = get_embeddings()
    try:
        if reuse:
            return load_vector_store(embeddings)
    except Exception:
        print("[Pipeline] No saved index found, building a new one...")
    return build_index()


def answer_question(question: str, vector_db: FAISS) -> str:
    """Run the full retrieve + generate flow for a question."""
    results = retrieve(vector_db, question)
    context = build_context(results)

    llm = get_llm()
    prompt = build_prompt(context, question)
    response = llm.invoke(prompt)
    return response.content