"""Step 6 – Configure the Groq LLM and prompt template."""

from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE

PROMPT_TEMPLATE = """You are Raju Tea Shop's AI assistant.

Answer the question using only the information provided below.

Information:
{context}

Question:
{question}

If the information is not available, say:
"I don't know based on the available information."

Answer:"""


def get_llm() -> ChatGroq:
    """Return a configured Groq chat model (OpenAI-compatible, free tier)."""
    print(f"[LLM] Using Groq model: {LLM_MODEL}")
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )


def build_prompt(context: str, question: str) -> str:
    """Fill the prompt template with context and question."""
    return PROMPT_TEMPLATE.format(context=context, question=question)