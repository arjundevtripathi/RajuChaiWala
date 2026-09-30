"""Colorful animated SaaS UI for Raju Tea Shop AI Assistant.

Run with:
    streamlit run main.py
"""

import time
import os
from pathlib import Path

import streamlit as st

from src.config import validate_config
from src.rag_pipeline import answer_question, get_index
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import build_vector_store
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader,
)
from langchain_core.documents import Document


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Raju Tea Shop · AI Assistant",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD CUSTOM CSS
# ============================================================
def load_css():
    css_file = Path(__file__).parent / "assets" / "styles.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

load_css()


# ============================================================
# FLOATING DECORATIVE EMOJIS
# ============================================================
st.markdown(
    """
    <div class="floating-emoji emoji-1">🍵</div>
    <div class="floating-emoji emoji-2">☕</div>
    <div class="floating-emoji emoji-3">🥟</div>
    <div class="floating-emoji emoji-4">✨</div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "messages": [],
    "vector_db": None,
    "active_source": "default",
    "uploaded_file_id": None,
    "stats": {"queries": 0, "avg_time": 0.0, "chunks": 0},
    "pending_question": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# FILE HANDLING
# ============================================================
def load_uploaded_file(uploaded_file) -> list[Document]:
    import tempfile

    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    suffix = os.path.splitext(name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        if name.endswith(".txt"):
            loader = TextLoader(tmp_path, encoding="utf-8")
        elif name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        elif name.endswith(".csv"):
            loader = CSVLoader(tmp_path)
        else:
            raise ValueError("Unsupported file. Use .txt, .pdf, .docx, or .csv")
        return loader.load()
    finally:
        os.unlink(tmp_path)


def build_index_from_documents(documents):
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    st.session_state.stats["chunks"] = len(chunks)
    return build_vector_store(chunks, embeddings, persist=False)


@st.cache_resource(show_spinner=False)
def load_default_index():
    validate_config()
    return get_index(reuse=True)


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">⚡ Powered by Groq · Llama 3.3 70B</div>
        <h1 class="hero-title">Raju Tea Shop<br/>AI Assistant</h1>
        <p class="hero-subtitle">
            Ask anything about the menu, prices, timings, offers, delivery, and more —
            grounded in Raju's real shop documents. Built with RAG, FAISS, and
            HuggingFace embeddings.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# METRIC CARDS
# ============================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Knowledge Base</div>
            <div class="metric-value accent">Raju's Shop</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Queries Answered</div>
            <div class="metric-value">{st.session_state.stats["queries"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    avg = st.session_state.stats["avg_time"]
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Avg. Response</div>
            <div class="metric-value">{avg:.2f}s</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📁 Knowledge Source")

    source_mode = st.radio(
        "Choose a source",
        ["Raju's default shop info", "Upload my own file"],
        index=0 if st.session_state.active_source == "default" else 1,
        label_visibility="collapsed",
    )

    uploaded_file = None
    if source_mode == "Upload my own file":
        uploaded_file = st.file_uploader(
            "Drop a document here",
            type=["txt", "pdf", "docx", "csv"],
            help="Supported: .txt, .pdf, .docx, .csv",
        )

    st.markdown("---")
    st.markdown("### ⚙️ Actions")

    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset stats", use_container_width=True):
        st.session_state.stats = {"queries": 0, "avg_time": 0.0, "chunks": 0}
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    samples = [
        "How much does masala tea cost?",
        "What time does the shop open?",
        "Is there a student discount?",
        "What snacks are available?",
        "Do you deliver?",
    ]
    for s in samples:
        if st.button(s, key=f"sample_{s}", use_container_width=True):
            st.session_state.pending_question = s
            st.rerun()

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit · LangChain · Groq")


# ============================================================
# SOURCE HANDLING
# ============================================================
def rebuild_for_upload(uploaded_file):
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.uploaded_file_id == file_id:
        return

    with st.spinner(f"Indexing {uploaded_file.name}..."):
        try:
            docs = load_uploaded_file(uploaded_file)
            vector_db = build_index_from_documents(docs)
        except Exception as exc:
            st.error(f"❌ Failed to index file: {exc}")
            st.stop()

    st.session_state.vector_db = vector_db
    st.session_state.active_source = "upload"
    st.session_state.uploaded_file_id = file_id
    st.session_state.messages = []


if source_mode == "Raju's default shop info":
    if st.session_state.active_source != "default":
        st.session_state.vector_db = load_default_index()
        st.session_state.active_source = "default"
        st.session_state.uploaded_file_id = None
        st.session_state.messages = []
        st.rerun()

    if st.session_state.vector_db is None:
        with st.spinner("Loading Raju's shop knowledge base..."):
            try:
                st.session_state.vector_db = load_default_index()
            except Exception as exc:
                st.error(f"❌ Failed to load knowledge base: {exc}")
                st.stop()
else:
    if uploaded_file is None:
        st.info("👈 Upload a `.txt`, `.pdf`, `.docx`, or `.csv` file to begin.")
        st.stop()
    rebuild_for_upload(uploaded_file)


# ============================================================
# STATUS
# ============================================================
if st.session_state.active_source == "default":
    st.success("📚 Using **Raju's default shop info** (`data/raju_shop.txt`)")
else:
    name = st.session_state.uploaded_file_id.split("_")[0]
    st.success(f"📄 Using **uploaded file**: `{name}`")


# ============================================================
# CHAT HISTORY
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🍵"):
        st.markdown(msg["content"])


# ============================================================
# ANSWER HANDLER
# ============================================================
def handle_question(question: str):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant", avatar="🍵"):
        with st.spinner("Brewing your answer..."):
            try:
                start = time.time()
                answer = answer_question(question, st.session_state.vector_db)
                elapsed = time.time() - start
            except Exception as exc:
                answer = f"⚠️ Error: {exc}"
                elapsed = 0.0

        st.markdown(answer)
        st.caption(f"⏱️ answered in {elapsed:.2f}s")

    st.session_state.messages.append({"role": "assistant", "content": answer})

    stats = st.session_state.stats
    total_time = stats["avg_time"] * stats["queries"] + elapsed
    stats["queries"] += 1
    stats["avg_time"] = total_time / stats["queries"]


# ============================================================
# SIDEBAR-TRIGGERED QUESTION
# ============================================================
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    handle_question(q)
    st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================
if prompt := st.chat_input("Ask a question about Raju Tea Shop..."):
    handle_question(prompt)