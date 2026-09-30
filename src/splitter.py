"""Step 2 – Split the document into chunks."""

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATOR


def split_documents(documents: list[Document]) -> list[Document]:
    """Split loaded documents into smaller chunks."""
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=CHUNK_SEPARATOR,
    )
    chunks = splitter.split_documents(documents)
    print(f"[Splitter] Created {len(chunks)} chunk(s)")
    return chunks


if __name__ == "__main__":
    from src.loader import load_document

    chunks = split_documents(load_document())
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk.page_content)