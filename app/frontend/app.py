"""Main Streamlit RAG AI chat application."""

import json
import uuid

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = "http://localhost:8000/api"
TIMEOUT = 60          # used for all non-streaming requests
STREAM_TIMEOUT = 300  # seconds; streaming can take a while on CPU

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { padding: 0rem 1rem; }
    .stChatMessage { padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def load_documents() -> list:
    try:
        r = requests.get(
            f"{API_BASE_URL}/documents/list",
            params={"user_id": st.session_state.user_id},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        st.session_state.documents = r.json()
        return st.session_state.documents
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
        return []


def upload_document(uploaded_file) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files={"file": uploaded_file},
            params={"user_id": st.session_state.user_id},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def delete_document(doc_id: int) -> bool:
    try:
        r = requests.delete(
            f"{API_BASE_URL}/documents/{doc_id}",
            params={"user_id": st.session_state.user_id},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        load_documents()
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


def get_query_history() -> list:
    try:
        r = requests.get(
            f"{API_BASE_URL}/queries/history",
            params={"user_id": st.session_state.user_id, "limit": 20},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return []


def stream_question(
    question: str,
    top_k: int,
    temperature: float,
):
    """
    Call POST /api/queries/ask-stream and yield tokens as they arrive.

    Also captures the sources and query_id from the SSE envelope and
    stores them in st.session_state for display after streaming ends.
    """
    payload = {
        "question": question,
        "user_id": st.session_state.user_id,
        "top_k": top_k,
        "temperature": temperature,
    }

    # Reset pending state
    st.session_state["_pending_chunks"] = []
    st.session_state["_pending_similarity"] = 0.0
    st.session_state["_pending_query_id"] = None

    with requests.post(
        f"{API_BASE_URL}/queries/ask-stream",
        json=payload,
        stream=True,
        timeout=STREAM_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])

            if event["type"] == "sources":
                st.session_state["_pending_chunks"] = event["retrieved_chunks"]
                st.session_state["_pending_similarity"] = event["top_similarity"]

            elif event["type"] == "token":
                yield event["content"]

            elif event["type"] == "done":
                st.session_state["_pending_query_id"] = event["query_id"]

            elif event["type"] == "error":
                raise RuntimeError(event.get("detail", "Unknown streaming error"))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("🤖 RAG AI Chat Assistant")
st.caption(f"Session: {st.session_state.user_id[:8]}...")

col1, col2 = st.columns([2, 1])

# ── Right column: documents + settings ─────────────────────────────────────
with col2:
    st.header("📁 Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md"],
        key="doc_uploader",
    )
    if uploaded_file:
        with st.spinner("Uploading and embedding…"):
            result = upload_document(uploaded_file)
            if result:
                st.success(
                    f"✅ {result['filename']} — {result['chunks_created']} chunks"
                )
                load_documents()

    st.divider()
    st.subheader("Your Documents")
    docs = load_documents()

    if docs:
        for doc in docs:
            c_doc, c_del = st.columns([4, 1])
            with c_doc:
                st.caption(f"📄 {doc['filename']}  ({doc['chunks']} chunks)")
            with c_del:
                if st.button("🗑️", key=f"del_{doc['id']}", help="Delete"):
                    delete_document(doc["id"])
                    st.rerun()
    else:
        st.info("No documents yet. Upload one to get started!")

    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", 1, 10, 5)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.1, step=0.05)

# ── Left column: chat ───────────────────────────────────────────────────────
with col1:
    st.header("💬 Chat")

    # Message history display
    chat_container = st.container(height=420, border=True)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("chunks"):
                    with st.expander("📚 Sources"):
                        for chunk in msg["chunks"]:
                            st.caption(
                                f"**{chunk['filename']}**  sim: {chunk['similarity']}"
                            )
                            st.text(chunk["text"])

    st.divider()

    # Input row
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Ask me anything…",
            placeholder="Type your question here",
            label_visibility="collapsed",
            key="user_input",
        )
    with col_send:
        send_button = st.button("Send", key="send_btn", use_container_width=True)

    # Process submission
    if send_button and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            try:
                # st.write_stream consumes our generator and renders tokens live
                full_answer = st.write_stream(
                    stream_question(user_input, top_k=top_k, temperature=temperature)
                )
            except Exception as e:
                full_answer = f"⚠️ Error: {e}"
                st.error(full_answer)

        # Persist the complete message (with sources) into session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_answer,
                "chunks": st.session_state.pop("_pending_chunks", []),
                "similarity": st.session_state.pop("_pending_similarity", 0.0),
                "query_id": st.session_state.pop("_pending_query_id", None),
            }
        )
        st.rerun()

    # Action buttons
    col_clear, col_hist = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_hist:
        if st.button("📜 Show History", use_container_width=True):
            st.session_state.show_history = not st.session_state.get(
                "show_history", False
            )

# ---------------------------------------------------------------------------
# Query history panel
# ---------------------------------------------------------------------------
if st.session_state.get("show_history", False):
    st.divider()
    st.subheader("📜 Query History")
    history = get_query_history()
    if history:
        for q in history:
            with st.expander(f"Q: {q['question'][:60]}… ({q['created_at'][:10]})"):
                st.write(f"**Answer:** {q['answer'][:300]}…")
                st.caption(f"{q['chunk_count']} chunks retrieved")
    else:
        st.info("No query history yet.")
