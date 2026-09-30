"""Step 1 – Load the knowledge base document."""

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from src.config import DATA_FILE


def load_document(path=DATA_FILE) -> list[Document]:
    """Load Raju's shop information from a text file."""
    loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()
    print(f"[Loader] Loaded {len(documents)} document(s) from {path}")
    return documents


if __name__ == "__main__":
    docs = load_document()
    print(docs[0].page_content[:200], "...")