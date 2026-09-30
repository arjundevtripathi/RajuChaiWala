"""Step 3 – Create the embedding model (local, no API key needed).

Groq does not provide embedding models, so we use a free, local
HuggingFace model instead.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a local HuggingFace embedding model."""
    print(f"[Embeddings] Loading local model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)