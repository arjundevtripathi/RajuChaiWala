"""Simple sanity tests for the RAG pipeline."""

from src.loader import load_document
from src.splitter import split_documents
from src.retriever import build_context


def test_load_document():
    docs = load_document()
    assert len(docs) >= 1
    assert "RAJU TEA SHOP" in docs[0].page_content


def test_split_documents():
    chunks = split_documents(load_document())
    assert len(chunks) >= 1
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_build_context():
    docs = load_document()
    chunks = split_documents(docs)
    context = build_context(chunks)
    assert "Masala Tea" in context