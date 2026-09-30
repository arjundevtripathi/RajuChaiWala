"""Central configuration for the RAG project."""

from pathlib import Path
import os

from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
VECTOR_STORE_DIR = OUTPUTS_DIR / "faiss_index"

DATA_FILE = DATA_DIR / "raju_shop.txt"

# ------------------------------------------------------------
# Chunking settings
# ------------------------------------------------------------
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
CHUNK_SEPARATOR = "\n"

# ------------------------------------------------------------
# Retrieval settings
# ------------------------------------------------------------
TOP_K = 2

# ------------------------------------------------------------
# LLM settings (Groq)
# ------------------------------------------------------------
LLM_MODEL = "openai/gpt-oss-120b"  # recommended default
LLM_TEMPERATURE = 0.0

# ------------------------------------------------------------
# Embeddings settings (HuggingFace local)
# ------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------------------------------------------
# API Key (Groq)
# ------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def validate_config() -> None:
    """Raise a helpful error if required config is missing."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example to .env and add your key, "
            "or export it in your shell. "
            "Get a free key at https://console.groq.com"
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge base file not found: {DATA_FILE}"
        )


# Ensure outputs directory exists
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)