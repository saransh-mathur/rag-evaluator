"""RAG AI Chat — full-featured Streamlit frontend."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL = "http://localhost:8000/api"
TIMEOUT = 60
STREAM_TIMEOUT = 300

EXAMPLE_QUESTIONS = [
    "📝  Summarize the key points of this document",
    "❓  What is the main topic covered?",
    "🔍  What are the most important facts mentioned?",
    "📊  Are there any numbers, dates, or statistics?",
]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG AI Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme CSS — navy + white  (+ dark/light toggle support)
# ---------------------------------------------------------------------------
DARK_CSS = """
:root {
    --bg:        #0a1628;
    --panel:     #0d1f3c;
    --panel2:    #112240;
    --border:    #1e3a5f;
    --border2:   #2a5298;
    --accent:    #4a9eff;
    --accent2:   #7ab3ef;
    --text:      #f0f4ff;
    --text-dim:  #a8c4e0;
    --text-mute: #6a8aaf;
    --warn:      #f0a500;
    --success:   #2ecc71;
    --error:     #e74c3c;
}
"""
LIGHT_CSS = """
:root {
    --bg:        #f0f4ff;
    --panel:     #ffffff;
    --panel2:    #e8eef8;
    --border:    #c0cce0;
    --border2:   #4a9eff;
    --accent:    #1a3a6c;
    --accent2:   #2a5298;
    --text:      #0a1628;
    --text-dim:  #2a4a6c;
    --text-mute: #5a7a9f;
    --warn:      #c0860a;
    --success:   #1a8a4a;
    --error:     #c0392b;
}
"""

SHARED_CSS = """
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stMain"] { background-color: var(--bg); }

[data-testid="stSidebar"] {
    background-color: var(--panel);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-dim) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text) !important; }

[data-testid="stChatMessage"] {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 8px;
    padding: 4px 8px;
}

[data-testid="stChatInput"] {
    background-color: var(--panel) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: var(--accent) !important; }

.stButton > button {
    background-color: var(--panel2);
    color: var(--text);
    border: 1px solid var(--border2);
    border-radius: 8px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background-color: var(--border2);
    border-color: var(--accent);
    color: #ffffff;
}

.plus-btn > button {
    background-color: var(--panel) !important;
    border: 2px solid var(--accent) !important;
    border-radius: 50% !important;
    color: var(--accent) !important;
    font-size: 1.4rem !important;
    height: 44px !important;
    width: 44px !important;
    padding: 0 !important;
}
.plus-btn > button:hover {
    background-color: var(--accent) !important;
    color: #ffffff !important;
}

.new-chat-btn > button {
    background: linear-gradient(135deg, var(--panel2), var(--border2)) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}

[data-testid="stExpander"] {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
}
[data-testid="stExpander"] summary { color: var(--accent2) !important; }

[data-testid="stFileUploader"] {
    background-color: var(--panel2);
    border: 1px dashed var(--border2);
    border-radius: 10px;
}

[data-testid="stDialog"] {
    background-color: var(--panel) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 16px !important;
}
[data-testid="stDialog"] * { color: var(--text) !important; }

[data-testid="stMetric"] {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] { color: var(--accent) !important; font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { color: var(--text-mute) !important; }

/* Source cards */
.source-card {
    background-color: var(--panel2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}
.source-card .src-header { color: var(--accent2); font-weight: 600; margin-bottom: 4px; }
.source-card .src-text   { color: var(--text-dim); line-height: 1.5; white-space: pre-wrap; }
.source-card .src-bar-bg {
    background-color: var(--border);
    border-radius: 4px;
    height: 6px;
    margin-top: 6px;
    overflow: hidden;
}
.source-card .src-bar-fill {
    height: 6px;
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background-color: var(--panel);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.85rem;
    color: var(--accent2);
    margin-bottom: 8px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

/* Typing dots */
.typing-dots {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 8px 4px;
}
.typing-dots span {
    width: 8px; height: 8px;
    background-color: var(--accent2);
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,80%,100%{transform:scale(0.6)} 40%{transform:scale(1)} }

/* Suggestion pills */
.suggestion-pill {
    display: inline-block;
    background-color: var(--panel2);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    color: var(--accent2);
    cursor: pointer;
    margin: 3px 4px 3px 0;
    transition: all 0.15s ease;
}
.suggestion-pill:hover {
    background-color: var(--border2);
    color: var(--text);
}

/* History cards */
.history-card {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.history-card .h-date  { color: var(--text-mute); font-size: 0.7rem; font-weight: 700;
                          text-transform: uppercase; letter-spacing: 0.05em; margin-bottom:8px; }
.history-card .h-q     { color: var(--accent2); font-weight: 600; margin-bottom: 5px; }
.history-card .h-a     { color: var(--text-dim); font-size: 0.88rem; line-height: 1.5; }
.history-card .h-meta  { color: var(--text-mute); font-size: 0.75rem; margin-top: 6px; }

/* Welcome card */
.welcome-card {
    background: linear-gradient(135deg, var(--panel2), var(--panel));
    border: 1px solid var(--border2);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 40px auto;
    max-width: 600px;
}
.welcome-card h2 { color: var(--text); margin-bottom: 8px; }
.welcome-card p  { color: var(--text-dim); margin-bottom: 20px; }

/* Doc chunk badge */
.chunk-badge-green  { color: #2ecc71; font-weight: 700; }
.chunk-badge-yellow { color: #f0a500; font-weight: 700; }
.chunk-badge-red    { color: #e74c3c; font-weight: 700; }

/* Feedback buttons */
.fb-row { display: flex; gap: 8px; margin-top: 6px; }
.fb-btn {
    background: none;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 3px 10px;
    cursor: pointer;
    font-size: 1rem;
    color: var(--text-mute);
    transition: all 0.15s;
}
.fb-btn:hover { border-color: var(--accent); color: var(--accent); }
.fb-btn.active-up   { border-color: #2ecc71; color: #2ecc71; background-color: var(--panel2); }
.fb-btn.active-down { border-color: #e74c3c; color: #e74c3c; background-color: var(--panel2); }

/* Copy button */
.copy-btn {
    float: right;
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px 8px;
    cursor: pointer;
    font-size: 0.75rem;
    color: var(--text-mute);
    transition: all 0.15s;
}
.copy-btn:hover { border-color: var(--accent); color: var(--accent); }

/* Timestamp */
.msg-ts { font-size: 0.7rem; color: var(--text-mute); margin-top: 4px; }

/* Empty doc state */
.empty-docs {
    text-align: center;
    padding: 20px 10px;
    color: var(--text-mute);
    font-size: 0.85rem;
    border: 1px dashed var(--border);
    border-radius: 10px;
}

/* Progress bar in upload modal */
.upload-progress {
    background-color: var(--border);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin: 8px 0;
}
.upload-progress-fill {
    height: 8px;
    border-radius: 6px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    animation: progress-anim 1.5s infinite;
    width: 60%;
}
@keyframes progress-anim {
    0%   { margin-left: -60%; }
    100% { margin-left: 100%; }
}

hr { border-color: var(--border) !important; }
h1 { color: var(--text) !important; letter-spacing: -0.5px; }
h2, h3 { color: var(--text-dim) !important; }
.stCaption, small { color: var(--text-mute) !important; }
[data-testid="stAlert"] { border-radius: 8px; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
"""

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
_defaults = {
    "user_id":      str(uuid.uuid4()),
    "messages":     [],
    "documents":    [],
    "show_history": False,
    "dark_mode":    True,
    "feedback":     {},   # query_id -> 1 or -1
    "suggestions":  [],   # current follow-up suggestions
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Inject CSS based on current mode
theme_vars = DARK_CSS if st.session_state.dark_mode else LIGHT_CSS
st.markdown(f"<style>{theme_vars}{SHARED_CSS}</style>", unsafe_allow_html=True)

# JS copy helper
st.markdown("""
<script>
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {}, function() {
        const ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta);
        ta.select(); document.execCommand('copy');
        document.body.removeChild(ta);
    });
}
</script>
""", unsafe_allow_html=True)


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
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
    return st.session_state.documents


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
        st.toast(f"❌ Upload failed: {e}", icon="🚫")
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
        st.toast(f"❌ Delete failed: {e}", icon="🚫")
        return False


def get_query_history() -> list:
    try:
        r = requests.get(
            f"{API_BASE_URL}/queries/history",
            params={"user_id": st.session_state.user_id, "limit": 50},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return []


def get_suggestions(question: str, answer: str) -> list[str]:
    try:
        r = requests.post(
            f"{API_BASE_URL}/queries/suggest",
            json={"question": question, "answer": answer},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("suggestions", [])
    except Exception:
        return []


def submit_feedback(query_id: int, feedback: int):
    try:
        requests.post(
            f"{API_BASE_URL}/queries/feedback",
            json={"query_id": query_id, "user_id": st.session_state.user_id,
                  "feedback": feedback},
            timeout=TIMEOUT,
        )
    except Exception:
        pass


def stream_question(question: str, top_k: int, temperature: float, max_tokens: int):
    payload = {
        "question":    question,
        "user_id":     st.session_state.user_id,
        "top_k":       top_k,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    st.session_state["_pending_chunks"]     = []
    st.session_state["_pending_similarity"] = 0.0
    st.session_state["_pending_query_id"]   = None

    with requests.post(
        f"{API_BASE_URL}/queries/ask-stream",
        json=payload, stream=True, timeout=STREAM_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "sources":
                st.session_state["_pending_chunks"]     = event["retrieved_chunks"]
                st.session_state["_pending_similarity"] = event["top_similarity"]
            elif event["type"] == "token":
                yield event["content"]
            elif event["type"] == "done":
                st.session_state["_pending_query_id"] = event["query_id"]
            elif event["type"] == "error":
                raise RuntimeError(event.get("detail", "Streaming error"))


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _chunk_badge(chunks: int) -> str:
    if chunks >= 10:
        return f'<span class="chunk-badge-green">● {chunks}</span>'
    elif chunks >= 3:
        return f'<span class="chunk-badge-yellow">● {chunks}</span>'
    return f'<span class="chunk-badge-red">● {chunks}</span>'


def render_sources(chunks: list):
    if not chunks:
        return
    with st.expander(f"📚 {len(chunks)} source chunk(s) used", expanded=False):
        for chunk in chunks:
            pct = int(chunk["similarity"] * 100)
            bar_color = (
                "#4a9eff" if pct >= 70 else
                "#f0a500" if pct >= 50 else
                "#6a8aaf"
            )
            st.markdown(
                f"""<div class="source-card">
                    <div class="src-header">📄 {chunk['filename']}</div>
                    <div class="src-text">{chunk['text']}</div>
                    <div class="src-bar-bg">
                        <div class="src-bar-fill"
                             style="width:{pct}%;background-color:{bar_color}"></div>
                    </div>
                    <small style="color:{bar_color}">Similarity {pct}%</small>
                </div>""",
                unsafe_allow_html=True,
            )


def render_message(msg: dict, idx: int):
    """Render a single chat message with timestamp, copy, and feedback."""
    with st.chat_message(msg["role"]):
        # Timestamp
        ts = msg.get("ts", "")
        if ts:
            st.markdown(f'<div class="msg-ts">{ts}</div>', unsafe_allow_html=True)

        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            qid = msg.get("query_id")

            # Copy button via JS
            escaped = msg["content"].replace("`", "\\`").replace("\n", "\\n")
            st.markdown(
                f"""<button class="copy-btn"
                    onclick="copyToClipboard(`{escaped}`);this.innerText='✅ Copied';
                    setTimeout(()=>this.innerText='📋 Copy',2000)">📋 Copy</button>""",
                unsafe_allow_html=True,
            )

            # Sources
            if msg.get("chunks"):
                render_sources(msg["chunks"])

            # Feedback
            if qid:
                current_fb = st.session_state.feedback.get(qid, 0)
                up_cls   = "active-up"   if current_fb ==  1 else ""
                down_cls = "active-down" if current_fb == -1 else ""
                fb_col1, fb_col2, _ = st.columns([1, 1, 10])
                with fb_col1:
                    if st.button("👍", key=f"up_{idx}_{qid}",
                                 help="Good answer", use_container_width=True):
                        st.session_state.feedback[qid] = 1
                        submit_feedback(qid, 1)
                        st.toast("Thanks for your feedback!", icon="👍")
                with fb_col2:
                    if st.button("👎", key=f"dn_{idx}_{qid}",
                                 help="Poor answer", use_container_width=True):
                        st.session_state.feedback[qid] = -1
                        submit_feedback(qid, -1)
                        st.toast("Thanks — we'll use this to improve.", icon="👎")


def render_welcome():
    """Welcome screen shown when no messages exist."""
    st.markdown(
        """<div class="welcome-card">
            <h2>👋 Welcome to RAG AI Chat</h2>
            <p>Upload a document using the <b>＋</b> button below, then ask me anything about it.<br>
            I'll find the most relevant sections and give you a detailed answer.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state["_prefill"] = q.split("  ", 1)[-1].strip()
                st.rerun()


def _group_history_by_date(history: list) -> dict[str, list]:
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    today = datetime.now(timezone.utc).date()
    for q in history:
        try:
            dt = datetime.fromisoformat(q["created_at"].replace("Z", "+00:00"))
            d = dt.date()
        except Exception:
            d = today
        diff = (today - d).days
        if diff == 0:
            label = "Today"
        elif diff == 1:
            label = "Yesterday"
        elif diff <= 7:
            label = "This week"
        else:
            label = d.strftime("%B %Y")
        groups[label].append(q)
    return dict(groups)


# ---------------------------------------------------------------------------
# Upload modal
# ---------------------------------------------------------------------------
@st.dialog("📎 Upload a Document", width="large")
def upload_modal():
    st.markdown(
        """<div style="background-color:var(--panel2);border:1px solid var(--border);
            border-radius:10px;padding:14px 18px;margin-bottom:16px;
            font-size:0.88rem;color:var(--text-dim);line-height:1.8">
        <b style="color:var(--accent2)">Supported formats</b><br>
        📄 <b>PDF</b> — text-based PDFs (not scanned images)<br>
        📝 <b>TXT</b> — plain text files<br>
        📋 <b>MD</b>  — Markdown documents<br><br>
        <b style="color:var(--accent2)">How it works</b><br>
        Your document is split into overlapping chunks → each chunk is embedded into
        a 768-dim vector → stored in PostgreSQL pgvector. When you ask a question,
        the top-K most similar chunks are retrieved and fed to the LLM as context.
        </div>""",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "md"],
        key="modal_uploader",
        help="Recommended max size: 10 MB",
    )

    if uploaded_file:
        size_kb = len(uploaded_file.getvalue()) / 1024
        # Document preview — first 500 chars of raw bytes
        try:
            preview = uploaded_file.getvalue()[:1000].decode("utf-8", errors="ignore")
            preview = preview[:400].strip()
        except Exception:
            preview = ""

        st.markdown(
            f"**{uploaded_file.name}** &nbsp;·&nbsp; "
            f"<span style='color:var(--text-mute)'>{size_kb:.0f} KB"
            f" · {uploaded_file.type or 'unknown'}</span>",
            unsafe_allow_html=True,
        )

        if preview:
            with st.expander("👁 Preview (first 400 chars)", expanded=False):
                st.markdown(
                    f"<pre style='font-size:0.8rem;color:var(--text-dim);"
                    f"white-space:pre-wrap'>{preview}…</pre>",
                    unsafe_allow_html=True,
                )

        if st.button("⬆️ Upload & Embed", use_container_width=True, type="primary"):
            progress_box = st.empty()
            progress_box.markdown(
                "<div class='upload-progress'><div class='upload-progress-fill'></div></div>"
                "<small style='color:var(--text-mute)'>Extracting and embedding…</small>",
                unsafe_allow_html=True,
            )
            result = upload_document(uploaded_file)
            progress_box.empty()
            if result:
                st.toast(
                    f"✅ {result['filename']} — {result['chunks_created']} chunks created",
                    icon="📄",
                )
                load_documents()
                st.success(
                    f"✅ **{result['filename']}** uploaded successfully — "
                    f"{result['chunks_created']} chunks embedded."
                )

    docs = st.session_state.documents
    if docs:
        st.divider()
        st.markdown("**Documents in this session**")
        for doc in docs:
            c1, c2 = st.columns([5, 1])
            chunks = doc["chunks"]
            badge = _chunk_badge(chunks)
            c1.markdown(
                f"<small>📄 <b>{doc['filename']}</b> &nbsp;"
                f"{badge} chunks &nbsp;"
                f"<span style='color:var(--text-mute)'>{doc['uploaded_at'][:10]}</span></small>",
                unsafe_allow_html=True,
            )
            if c2.button("✕", key=f"modal_del_{doc['id']}", help="Remove"):
                if delete_document(doc["id"]):
                    st.toast(f"Deleted {doc['filename']}", icon="🗑️")
                st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── New Chat (top, prominent) ─────────────────────────────────────
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True, key="new_chat_top"):
        st.session_state.user_id    = str(uuid.uuid4())
        st.session_state.messages   = []
        st.session_state.documents  = []
        st.session_state.suggestions = []
        st.session_state.show_history = False
        st.session_state.feedback   = {}
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 RAG AI Chat")
    st.caption(f"Session `{st.session_state.user_id[:8]}…`")
    st.divider()

    # ── Documents ────────────────────────────────────────────────────
    docs = load_documents()
    doc_count = len(docs)
    st.markdown(
        f"**📁 Documents** &nbsp;"
        f"<span style='color:var(--accent);font-size:0.8rem'>"
        f"({doc_count} file{'s' if doc_count != 1 else ''})</span>",
        unsafe_allow_html=True,
    )
    if docs:
        for doc in docs:
            c_name, c_del = st.columns([5, 1])
            chunks = doc["chunks"]
            badge = _chunk_badge(chunks)
            with c_name:
                st.markdown(
                    f"<small>📄 <b>{doc['filename']}</b><br>"
                    f"{badge} <span style='color:var(--text-mute)'>chunks</span></small>",
                    unsafe_allow_html=True,
                )
            with c_del:
                if st.button("✕", key=f"side_del_{doc['id']}", help="Delete"):
                    if delete_document(doc["id"]):
                        st.toast(f"Deleted {doc['filename']}", icon="🗑️")
                    st.rerun()
    else:
        st.markdown(
            "<div class='empty-docs'>📂<br>No documents yet<br>"
            "<small>Use ＋ in the chat bar to upload</small></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Settings (collapsible) ───────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=False):
        top_k = st.slider("Chunks to retrieve", 1, 12, 8,
                          help="More = richer context, slower retrieval")
        temperature = st.slider("Creativity", 0.0, 1.0, 0.1, step=0.05,
                                help="0 = focused, 1 = creative")
        max_tokens = st.slider("Max response length", 256, 4096, 2048, step=256,
                               help="Maximum tokens the LLM will generate")

        st.divider()

        # Dark / Light mode toggle
        mode_label = "☀️ Light mode" if st.session_state.dark_mode else "🌙 Dark mode"
        if st.button(mode_label, use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.divider()

    # ── Clear History toggle ─────────────────────────────────────────
    hist_label = "📜 Hide History" if st.session_state.show_history else "📜 View History"
    if st.button(hist_label, use_container_width=True):
        st.session_state.show_history = not st.session_state.show_history
        st.rerun()

    # ── Session stats ────────────────────────────────────────────────
    if st.session_state.messages:
        st.divider()
        q_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
        fb_pos = sum(1 for v in st.session_state.feedback.values() if v == 1)
        fb_neg = sum(1 for v in st.session_state.feedback.values() if v == -1)
        c1, c2 = st.columns(2)
        c1.metric("Queries", q_count)
        c2.metric("Docs", doc_count)
        if fb_pos or fb_neg:
            st.caption(f"Feedback: 👍 {fb_pos}  👎 {fb_neg}")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='margin-bottom:2px'>💬 RAG AI Chat</h1>"
    "<p style='color:var(--text-mute);margin-top:0;margin-bottom:16px;font-size:0.95rem'>"
    "Ask questions about your documents — answers stream in real time</p>",
    unsafe_allow_html=True,
)

# ── Interaction history panel ────────────────────────────────────────────
if st.session_state.show_history:
    st.markdown("### 📜 Interaction History")
    history = get_query_history()
    if history:
        groups = _group_history_by_date(history)
        for group_label, items in groups.items():
            st.markdown(
                f"<div style='color:var(--text-mute);font-size:0.72rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.07em;"
                f"margin:16px 0 8px'>{group_label}</div>",
                unsafe_allow_html=True,
            )
            for q in items:
                ans_preview = q["answer"][:300].replace("\n", " ")
                if len(q["answer"]) > 300:
                    ans_preview += "…"
                ts = q["created_at"][:16].replace("T", " ")
                fb_icon = (
                    " 👍" if q.get("id") and st.session_state.feedback.get(q["id"]) == 1 else
                    " 👎" if q.get("id") and st.session_state.feedback.get(q["id"]) == -1 else ""
                )
                st.markdown(
                    f"""<div class="history-card">
                        <div class="h-q">Q: {q['question']}{fb_icon}</div>
                        <div class="h-a">{ans_preview}</div>
                        <div class="h-meta">🕐 {ts} &nbsp;·&nbsp; 📚 {q['chunk_count']} chunks</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        st.divider()
    else:
        st.info("No interaction history yet.")

# ── Welcome or messages ──────────────────────────────────────────────────
if not st.session_state.messages:
    render_welcome()
else:
    for idx, msg in enumerate(st.session_state.messages):
        render_message(msg, idx)

# ── Follow-up suggestions ────────────────────────────────────────────────
if st.session_state.suggestions:
    st.markdown(
        "<small style='color:var(--text-mute)'>💡 Follow-up suggestions:</small>",
        unsafe_allow_html=True,
    )
    sug_cols = st.columns(len(st.session_state.suggestions))
    for i, sug in enumerate(st.session_state.suggestions):
        with sug_cols[i]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state["_prefill"] = sug
                st.session_state.suggestions = []
                st.rerun()

# ── Input row: [＋] + chat input ─────────────────────────────────────────
in_col, chat_col = st.columns([1, 20])
with in_col:
    st.markdown('<div class="plus-btn">', unsafe_allow_html=True)
    if st.button("＋", key="open_upload", help="Upload a document"):
        upload_modal()
    st.markdown('</div>', unsafe_allow_html=True)

with chat_col:
    # Support pre-fill from example / suggestion click
    prefill = st.session_state.pop("_prefill", None)
    user_input = st.chat_input(
        "Ask me anything about your documents…",
        key="chat_input",
    )
    # If a suggestion/example was clicked, treat it as input
    if prefill and not user_input:
        user_input = prefill

# ── Handle submission ────────────────────────────────────────────────────
if user_input:
    st.session_state.suggestions = []
    ts_now = datetime.now().strftime("%H:%M")

    with st.chat_message("user"):
        st.markdown(f'<div class="msg-ts">{ts_now}</div>', unsafe_allow_html=True)
        st.markdown(user_input)
    st.session_state.messages.append({
        "role": "user", "content": user_input, "ts": ts_now
    })

    with st.chat_message("assistant"):
        status_box = st.empty()
        answer_box = st.empty()

        # Typing indicator while retrieval runs
        status_box.markdown(
            "<div class='status-badge'>🔍 Searching documents…</div>",
            unsafe_allow_html=True,
        )

        full_answer = ""
        sources_received = False

        try:
            for token in stream_question(
                user_input, top_k=top_k,
                temperature=temperature, max_tokens=max_tokens,
            ):
                if not sources_received:
                    sources_received = True
                    status_box.markdown(
                        "<div class='status-badge'>✍️ Generating answer…</div>",
                        unsafe_allow_html=True,
                    )
                full_answer += token
                answer_box.markdown(full_answer + "▌")

            answer_box.markdown(full_answer)
            status_box.empty()

        except Exception as e:
            status_box.empty()
            full_answer = f"⚠️ Error: {e}"
            answer_box.markdown(full_answer)

        chunks  = st.session_state.pop("_pending_chunks", [])
        qid     = st.session_state.pop("_pending_query_id", None)
        sim     = st.session_state.pop("_pending_similarity", 0.0)

        # Copy button
        escaped = full_answer.replace("`", "\\`").replace("\n", "\\n")
        st.markdown(
            f"""<button class="copy-btn"
                onclick="copyToClipboard(`{escaped}`);this.innerText='✅ Copied';
                setTimeout(()=>this.innerText='📋 Copy',2000)">📋 Copy</button>""",
            unsafe_allow_html=True,
        )

        # Sources
        render_sources(chunks)

        # Timestamp on answer
        st.markdown(f'<div class="msg-ts">{ts_now}</div>', unsafe_allow_html=True)

        # Feedback buttons
        if qid:
            fb_col1, fb_col2, _ = st.columns([1, 1, 10])
            with fb_col1:
                if st.button("👍", key=f"up_new_{qid}", help="Good answer",
                             use_container_width=True):
                    st.session_state.feedback[qid] = 1
                    submit_feedback(qid, 1)
                    st.toast("Thanks for the feedback!", icon="👍")
            with fb_col2:
                if st.button("👎", key=f"dn_new_{qid}", help="Poor answer",
                             use_container_width=True):
                    st.session_state.feedback[qid] = -1
                    submit_feedback(qid, -1)
                    st.toast("Thanks — noted for improvement.", icon="👎")

    # Persist message
    st.session_state.messages.append({
        "role":     "assistant",
        "content":  full_answer,
        "chunks":   chunks,
        "similarity": sim,
        "query_id": qid,
        "ts":       ts_now,
    })

    # Fetch follow-up suggestions in background (non-blocking feel)
    if full_answer and not full_answer.startswith("⚠️"):
        sugs = get_suggestions(user_input, full_answer)
        if sugs:
            st.session_state.suggestions = sugs
            st.rerun()
