"""RAG AI Chat — full-featured Streamlit frontend (v2)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE_URL   = "http://localhost:8000/api"
TIMEOUT        = 60
STREAM_TIMEOUT = 300

EXAMPLE_QUESTIONS = [
    "📝  Summarize the key points of this document",
    "❓  What is the main topic covered?",
    "🔍  What are the most important facts mentioned?",
    "📊  Are there any numbers, dates, or statistics?",
]

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG AI Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
DARK_CSS = """:root {
    --bg:#0a1628;--panel:#0d1f3c;--panel2:#112240;
    --border:#1e3a5f;--border2:#2a5298;
    --accent:#4a9eff;--accent2:#7ab3ef;
    --text:#f0f4ff;--text-dim:#a8c4e0;--text-mute:#6a8aaf;
    --warn:#f0a500;--success:#2ecc71;--error:#e74c3c;
}"""
LIGHT_CSS = """:root {
    --bg:#f0f4ff;--panel:#ffffff;--panel2:#e8eef8;
    --border:#c0cce0;--border2:#4a9eff;
    --accent:#1a3a6c;--accent2:#2a5298;
    --text:#0a1628;--text-dim:#2a4a6c;--text-mute:#5a7a9f;
    --warn:#c0860a;--success:#1a8a4a;--error:#c0392b;
}"""

SHARED_CSS = """
html,body,[data-testid="stAppViewContainer"]{background:var(--bg);color:var(--text);font-family:'Inter','Segoe UI',sans-serif;}
[data-testid="stMain"]{background:var(--bg);}
[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text-dim)!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:var(--text)!important;}
[data-testid="stChatMessage"]{background:var(--panel);border:1px solid var(--border);border-radius:12px;margin-bottom:8px;padding:4px 8px;}
[data-testid="stChatInput"]{background:var(--panel)!important;border:1px solid var(--border2)!important;border-radius:12px!important;}
[data-testid="stChatInput"] textarea{color:var(--text)!important;}
[data-testid="stChatInputSubmitButton"] svg{fill:var(--accent)!important;}
.stButton>button{background:var(--panel2);color:var(--text);border:1px solid var(--border2);border-radius:8px;transition:all .2s;}
.stButton>button:hover{background:var(--border2);border-color:var(--accent);color:#fff;}
.plus-btn>button{background:var(--panel)!important;border:2px solid var(--accent)!important;border-radius:50%!important;color:var(--accent)!important;font-size:1.4rem!important;height:44px!important;width:44px!important;padding:0!important;}
.plus-btn>button:hover{background:var(--accent)!important;color:#fff!important;}
.new-chat-btn>button{background:linear-gradient(135deg,var(--panel2),var(--border2))!important;border:1px solid var(--accent)!important;border-radius:10px!important;color:var(--text)!important;font-weight:600!important;}
[data-testid="stExpander"]{background:var(--panel);border:1px solid var(--border);border-radius:8px;}
[data-testid="stExpander"] summary{color:var(--accent2)!important;}
[data-testid="stFileUploader"]{background:var(--panel2);border:1px dashed var(--border2);border-radius:10px;}
[data-testid="stDialog"]{background:var(--panel)!important;border:1px solid var(--border2)!important;border-radius:16px!important;}
[data-testid="stDialog"] *{color:var(--text)!important;}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px 16px;}
[data-testid="stMetricValue"]{color:var(--accent)!important;font-size:1.4rem!important;}
[data-testid="stMetricLabel"]{color:var(--text-mute)!important;}
.source-card{background:var(--panel2);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:.85rem;}
.source-card .src-header{color:var(--accent2);font-weight:600;margin-bottom:4px;}
.source-card .src-text{color:var(--text-dim);line-height:1.5;white-space:pre-wrap;}
.source-card .src-bar-bg{background:var(--border);border-radius:4px;height:6px;margin-top:6px;overflow:hidden;}
.source-card .src-bar-fill{height:6px;border-radius:4px;transition:width .4s;}
.status-badge{display:inline-flex;align-items:center;gap:8px;background:var(--panel);border:1px solid var(--border2);border-radius:20px;padding:6px 14px;font-size:.85rem;color:var(--accent2);margin-bottom:8px;animation:pulse 1.5s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
.history-card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px 16px;margin-bottom:10px;}
.history-card .h-q{color:var(--accent2);font-weight:600;margin-bottom:5px;}
.history-card .h-a{color:var(--text-dim);font-size:.88rem;line-height:1.5;}
.history-card .h-meta{color:var(--text-mute);font-size:.75rem;margin-top:6px;}
.welcome-card{background:linear-gradient(135deg,var(--panel2),var(--panel));border:1px solid var(--border2);border-radius:16px;padding:28px 32px;text-align:center;margin:40px auto;max-width:600px;}
.welcome-card h2{color:var(--text);margin-bottom:8px;}
.welcome-card p{color:var(--text-dim);margin-bottom:20px;}
.chunk-badge-green{color:#2ecc71;font-weight:700;}
.chunk-badge-yellow{color:#f0a500;font-weight:700;}
.chunk-badge-red{color:#e74c3c;font-weight:700;}
.copy-btn{float:right;background:none;border:1px solid var(--border);border-radius:6px;padding:2px 8px;cursor:pointer;font-size:.75rem;color:var(--text-mute);transition:all .15s;}
.copy-btn:hover{border-color:var(--accent);color:var(--accent);}
.msg-ts{font-size:.7rem;color:var(--text-mute);margin-top:4px;}
.star-active{color:#f0a500!important;}
.empty-docs{text-align:center;padding:20px 10px;color:var(--text-mute);font-size:.85rem;border:1px dashed var(--border);border-radius:10px;}
.upload-progress{background:var(--border);border-radius:6px;height:8px;overflow:hidden;margin:8px 0;}
.upload-progress-fill{height:8px;border-radius:6px;background:linear-gradient(90deg,var(--accent),var(--accent2));animation:prog 1.5s infinite;width:60%;}
@keyframes prog{0%{margin-left:-60%}100%{margin-left:100%}}
.summary-box{background:var(--panel2);border-left:3px solid var(--accent2);border-radius:6px;padding:8px 12px;font-size:.82rem;color:var(--text-dim);margin-top:4px;font-style:italic;}
hr{border-color:var(--border)!important;}
h1{color:var(--text)!important;letter-spacing:-.5px;}
h2,h3{color:var(--text-dim)!important;}
.stCaption,small{color:var(--text-mute)!important;}
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:var(--accent);}
#MainMenu,footer,[data-testid="stToolbar"]{visibility:hidden;}
"""

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_defaults: dict = {
    "user_id":       str(uuid.uuid4()),
    "messages":      [],
    "documents":     [],
    "show_history":  False,
    "dark_mode":     True,
    "feedback":      {},
    "starred":       set(),
    "suggestions":   [],
    "doc_mode":      True,
    "use_hyde":      False,
    "use_rerank":    True,
    "active_collection": "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

theme_vars = DARK_CSS if st.session_state.dark_mode else LIGHT_CSS
st.markdown(f"<style>{theme_vars}{SHARED_CSS}</style>", unsafe_allow_html=True)

st.markdown("""<script>
function copyFromElement(btnEl) {
    var id  = btnEl.getAttribute('data-copy-id');
    var src = document.getElementById(id);
    var txt = src ? src.textContent : '';
    navigator.clipboard.writeText(txt).catch(function() {
        var ta = document.createElement('textarea');
        ta.value = txt;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    });
    btnEl.innerText = '✅ Copied';
    setTimeout(function(){ btnEl.innerText = '📋 Copy'; }, 2000);
}
</script>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def load_documents() -> list:
    try:
        params: dict = {"user_id": st.session_state.user_id}
        if st.session_state.active_collection:
            params["collection"] = st.session_state.active_collection
        r = requests.get(f"{API_BASE_URL}/documents/list", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        st.session_state.documents = r.json()
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
    return st.session_state.documents


def load_collections() -> list[str]:
    try:
        r = requests.get(
            f"{API_BASE_URL}/documents/collections",
            params={"user_id": st.session_state.user_id},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("collections", [])
    except Exception:
        return []


def upload_document(uploaded_file, collection: str = "") -> dict | None:
    try:
        params: dict = {"user_id": st.session_state.user_id}
        if collection:
            params["collection"] = collection
        r = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files={"file": uploaded_file},
            params=params,
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


def search_document_api(doc_id: int, query: str) -> list:
    try:
        r = requests.get(
            f"{API_BASE_URL}/documents/{doc_id}/search",
            params={"user_id": st.session_state.user_id, "q": query},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []


def get_query_history(starred_only: bool = False) -> list:
    try:
        params: dict = {"user_id": st.session_state.user_id, "limit": 50}
        if starred_only:
            params["starred"] = "true"
        r = requests.get(f"{API_BASE_URL}/queries/history", params=params, timeout=TIMEOUT)
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


def get_autocomplete(partial: str) -> list[str]:
    try:
        r = requests.get(
            f"{API_BASE_URL}/queries/autocomplete",
            params={"user_id": st.session_state.user_id, "q": partial},
            timeout=5,
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


def submit_star(query_id: int, starred: bool):
    try:
        requests.post(
            f"{API_BASE_URL}/queries/star",
            json={"query_id": query_id, "user_id": st.session_state.user_id,
                  "starred": starred},
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
        "doc_mode":    st.session_state.doc_mode,
        "use_hyde":    st.session_state.use_hyde,
        "use_rerank":  st.session_state.use_rerank,
        # pass last 6 turns as chat history for multi-turn context
        "chat_history": [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[-6:]
            if m["role"] in ("user", "assistant")
        ],
    }
    st.session_state["_pending_chunks"]     = []
    st.session_state["_pending_similarity"] = 0.0
    st.session_state["_pending_query_id"]   = None
    st.session_state["_pending_token_usage"] = None

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
                st.session_state["_pending_token_usage"] = event.get("token_usage")
            elif event["type"] == "error":
                raise RuntimeError(event.get("detail", "Streaming error"))


def export_chat_md() -> str:
    """Generate a Markdown export of the current conversation."""
    lines = ["# RAG AI Chat Export\n",
             f"Session: `{st.session_state.user_id[:8]}…`  \n",
             f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n---\n"]
    for msg in st.session_state.messages:
        role = "**You**" if msg["role"] == "user" else "**Assistant**"
        ts   = msg.get("ts", "")
        lines.append(f"{role} _{ts}_\n\n{msg['content']}\n\n---\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _chunk_badge(n: int) -> str:
    if n >= 10: return f'<span class="chunk-badge-green">● {n}</span>'
    if n >= 3:  return f'<span class="chunk-badge-yellow">● {n}</span>'
    return       f'<span class="chunk-badge-red">● {n}</span>'


def render_sources(chunks: list):
    if not chunks:
        return
    with st.expander(f"📚 {len(chunks)} source chunk(s) used", expanded=False):
        for chunk in chunks:
            pct   = int(chunk["similarity"] * 100)
            color = "#4a9eff" if pct >= 70 else "#f0a500" if pct >= 50 else "#6a8aaf"
            st.markdown(
                f"""<div class="source-card">
                    <div class="src-header">📄 {chunk['filename']}</div>
                    <div class="src-text">{chunk['text']}</div>
                    <div class="src-bar-bg">
                      <div class="src-bar-fill" style="width:{pct}%;background:{color}"></div>
                    </div>
                    <small style="color:{color}">Similarity {pct}%</small>
                </div>""",
                unsafe_allow_html=True,
            )


def render_message(msg: dict, idx: int):
    with st.chat_message(msg["role"]):
        ts = msg.get("ts", "")
        if ts:
            st.markdown(f'<div class="msg-ts">{ts}</div>', unsafe_allow_html=True)
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            qid = msg.get("query_id")
            
            usage = msg.get("token_usage")
            if usage:
                st.markdown(
                    f"<div style='font-size:0.75rem;color:var(--text-mute);margin-bottom:8px;'>"
                    f"⚡ Tokens: <b>{usage.get('total_tokens', 0)}</b> "
                    f"(Prompt: {usage.get('prompt_tokens', 0)} | "
                    f"Completion: {usage.get('completion_tokens', 0)})</div>",
                    unsafe_allow_html=True,
                )

            # Copy — store content in hidden span, read from JS (avoids escaping issues)
            copy_id = f"copy_src_{idx}"
            import html as _html
            safe_content = _html.escape(msg["content"])
            st.markdown(
                f"""<span id="{copy_id}" style="display:none">{safe_content}</span>
                <button class="copy-btn" data-copy-id="{copy_id}"
                    onclick="copyFromElement(this)">📋 Copy</button>""",
                unsafe_allow_html=True,
            )

            # Sources
            if msg.get("chunks"):
                render_sources(msg["chunks"])

            # Feedback + star
            if qid:
                c1, c2, c3, _ = st.columns([1, 1, 1, 9])
                with c1:
                    if st.button("👍", key=f"up_{idx}_{qid}", use_container_width=True):
                        st.session_state.feedback[qid] = 1
                        submit_feedback(qid, 1)
                        st.toast("Thanks!", icon="👍")
                with c2:
                    if st.button("👎", key=f"dn_{idx}_{qid}", use_container_width=True):
                        st.session_state.feedback[qid] = -1
                        submit_feedback(qid, -1)
                        st.toast("Noted.", icon="👎")
                with c3:
                    is_starred = qid in st.session_state.starred
                    star_icon  = "⭐" if is_starred else "☆"
                    if st.button(star_icon, key=f"star_{idx}_{qid}", use_container_width=True,
                                 help="Pin this answer"):
                        if is_starred:
                            st.session_state.starred.discard(qid)
                            submit_star(qid, False)
                        else:
                            st.session_state.starred.add(qid)
                            submit_star(qid, True)
                        st.rerun()


def render_welcome():
    st.markdown(
        """<div class="welcome-card">
            <h2>👋 Welcome to RAG AI Chat</h2>
            <p>Upload a document with <b>＋</b>, then ask anything about it.<br>
            Answers stream in real time with sources shown.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("#### 💡 Try asking:")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                st.session_state["_prefill"] = q.split("  ", 1)[-1].strip()
                st.rerun()


def _group_history(history: list) -> dict[str, list]:
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    today = datetime.now(timezone.utc).date()
    for q in history:
        try:
            d = datetime.fromisoformat(q["created_at"].replace("Z", "+00:00")).date()
        except Exception:
            d = today
        diff = (today - d).days
        label = "Today" if diff == 0 else "Yesterday" if diff == 1 \
            else "This week" if diff <= 7 else d.strftime("%B %Y")
        groups[label].append(q)
    return dict(groups)


# ---------------------------------------------------------------------------
# Upload modal
# ---------------------------------------------------------------------------
@st.dialog("📎 Upload a Document", width="large")
def upload_modal():
    st.markdown(
        """<div style="background:var(--panel2);border:1px solid var(--border);
            border-radius:10px;padding:14px 18px;margin-bottom:16px;
            font-size:.88rem;color:var(--text-dim);line-height:1.8">
        <b style="color:var(--accent2)">Supported formats</b><br>
        📄 <b>PDF</b> — text/scanned PDFs &nbsp; 📝 <b>TXT/MD</b> — plain text &nbsp; 🖼️ <b>Image</b> — PNG, JPG (OCR)<br><br>
        <b style="color:var(--accent2)">How it works</b><br>
        Text is extracted → split into overlapping chunks → embedded into 768-dim vectors
        → stored in PostgreSQL pgvector. Retrieval uses hybrid BM25 + vector search with
        TF-IDF re-ranking. An AI summary is generated automatically after upload.
        </div>""",
        unsafe_allow_html=True,
    )

    col_file, col_col = st.columns([3, 2])
    with col_file:
        uploaded_file = st.file_uploader(
            "Choose a file", type=["pdf", "txt", "md", "png", "jpg", "jpeg"], key="modal_uploader"
        )
    with col_col:
        collections = load_collections()
        col_input = st.text_input(
            "Collection / tag (optional)",
            placeholder="e.g. research, project-a",
            key="upload_collection",
        )

    if uploaded_file:
        size_kb = len(uploaded_file.getvalue()) / 1024
        is_image = uploaded_file.name.lower().endswith((".png", ".jpg", ".jpeg"))
        
        try:
            if not is_image:
                preview = uploaded_file.getvalue()[:1000].decode("utf-8", errors="ignore")[:400].strip()
            else:
                preview = ""
        except Exception:
            preview = ""

        st.markdown(
            f"**{uploaded_file.name}** &nbsp;·&nbsp; "
            f"<span style='color:var(--text-mute)'>{size_kb:.0f} KB</span>",
            unsafe_allow_html=True,
        )
        
        if is_image:
            with st.expander("👁 Preview", expanded=False):
                st.image(uploaded_file, use_container_width=True)
        elif preview:
            with st.expander("👁 Preview (first 400 chars)", expanded=False):
                st.markdown(
                    f"<pre style='font-size:.8rem;color:var(--text-dim);white-space:pre-wrap'>{preview}…</pre>",
                    unsafe_allow_html=True,
                )

        if st.button("⬆️ Upload & Embed", use_container_width=True, type="primary"):
            pb = st.empty()
            pb.markdown(
                "<div class='upload-progress'><div class='upload-progress-fill'></div></div>"
                "<small>Extracting and embedding…</small>",
                unsafe_allow_html=True,
            )
            result = upload_document(uploaded_file, collection=col_input)
            pb.empty()
            if result:
                st.toast(f"✅ {result['filename']} — {result['chunks_created']} chunks", icon="📄")
                load_documents()
                st.success(
                    f"✅ **{result['filename']}** — {result['chunks_created']} chunks embedded.  \n"
                    f"🤖 Summary generating in background…"
                )

    # Doc list in modal
    docs = st.session_state.documents
    if docs:
        st.divider()
        st.markdown("**Documents in this session**")
        for doc in docs:
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                f"<small>📄 <b>{doc['filename']}</b> &nbsp;"
                f"{_chunk_badge(doc['chunks'])} chunks &nbsp;"
                f"<span style='color:var(--text-mute)'>{doc['uploaded_at'][:10]}</span></small>",
                unsafe_allow_html=True,
            )
            if doc.get("summary"):
                c1.markdown(
                    f'<div class="summary-box">{doc["summary"]}</div>',
                    unsafe_allow_html=True,
                )
            if c2.button("✕", key=f"mdel_{doc['id']}"):
                delete_document(doc["id"])
                st.toast(f"Deleted {doc['filename']}", icon="🗑️")
                st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True, key="new_chat"):
        for k, v in _defaults.items():
            st.session_state[k] = v if not callable(v) else v()
        st.session_state.user_id = str(uuid.uuid4())
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 RAG AI Chat")
    st.caption(f"Session `{st.session_state.user_id[:8]}…`")
    st.divider()

    # ── Collections filter ───────────────────────────────────────────
    collections = load_collections()
    all_cols = ["All"] + collections
    selected = st.selectbox(
        "📂 Collection", all_cols,
        index=0,
        key="col_select",
        help="Filter documents by collection"
    )
    st.session_state.active_collection = "" if selected == "All" else selected

    # ── Documents ────────────────────────────────────────────────────
    docs = load_documents()
    doc_count = len(docs)
    st.markdown(
        f"**📁 Documents** &nbsp;<span style='color:var(--accent);font-size:.8rem'>"
        f"({doc_count})</span>",
        unsafe_allow_html=True,
    )

    if docs:
        for doc in docs:
            c_n, c_d = st.columns([5, 1])
            with c_n:
                st.markdown(
                    f"<small>📄 <b>{doc['filename']}</b><br>"
                    f"{_chunk_badge(doc['chunks'])} "
                    f"<span style='color:var(--text-mute)'>chunks</span>"
                    + (f"<br><span style='color:var(--text-mute);font-size:.7rem'>"
                       f"{doc.get('collection_tag','')}</span>"
                       if doc.get('collection_tag') else "")
                    + "</small>",
                    unsafe_allow_html=True,
                )
                # Show summary if available
                if doc.get("summary"):
                    with st.expander("💡 Summary", expanded=False):
                        st.caption(doc["summary"])
            with c_d:
                if st.button("✕", key=f"sdel_{doc['id']}"):
                    delete_document(doc["id"])
                    st.toast(f"Deleted {doc['filename']}", icon="🗑️")
                    st.rerun()
    else:
        st.markdown(
            "<div class='empty-docs'>📂<br>No documents<br>"
            "<small>Use ＋ to upload</small></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Document search ──────────────────────────────────────────────
    if docs:
        with st.expander("🔍 Search within document", expanded=False):
            search_doc = st.selectbox(
                "Document", [d["filename"] for d in docs], key="search_doc_sel"
            )
            search_q = st.text_input("Keyword", key="doc_search_q")
            if search_q and search_doc:
                doc_id = next(d["id"] for d in docs if d["filename"] == search_doc)
                hits   = search_document_api(doc_id, search_q)
                if hits:
                    for h in hits:
                        st.markdown(
                            f"<small><b>Chunk {h['chunk_index']}</b>: "
                            f"{h['text'][:200]}…</small>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No matches found.")

    # ── Settings ────────────────────────────────────────────────────
    with st.expander("⚙️ Settings", expanded=False):
        import os
        default_top_k = int(os.getenv("TOP_K", 3))
        top_k       = st.slider("Chunks to retrieve",    1, 12, default_top_k)
        temperature = st.slider("Creativity",            0.0, 1.0, 0.1, step=0.05)
        max_tokens  = st.slider("Max response length",   256, 4096, 2048, step=256)

        st.divider()

        # Document mode toggle
        st.session_state.doc_mode = st.toggle(
            "📄 Document mode",
            value=st.session_state.doc_mode,
            help="ON = answer from your documents only. OFF = general LLM mode.",
        )

        # HyDE
        st.session_state.use_hyde = st.toggle(
            "🧠 HyDE query expansion",
            value=st.session_state.use_hyde,
            help="Expand query to hypothetical answer before embedding (improves recall for vague questions)",
        )

        # Re-rank
        st.session_state.use_rerank = st.toggle(
            "🔁 TF-IDF re-ranking",
            value=st.session_state.use_rerank,
            help="Re-score retrieved chunks by lexical overlap (combines with vector score)",
        )

        st.divider()

        mode_label = "☀️ Light mode" if st.session_state.dark_mode else "🌙 Dark mode"
        if st.button(mode_label, use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.divider()

    # ── History / starred ────────────────────────────────────────────
    hc1, hc2 = st.columns(2)
    with hc1:
        hist_lbl = "📜 Hide" if st.session_state.show_history else "📜 History"
        if st.button(hist_lbl, use_container_width=True):
            st.session_state.show_history = not st.session_state.show_history
            st.rerun()
    with hc2:
        if st.button("⭐ Starred", use_container_width=True):
            st.session_state["_show_starred"] = not st.session_state.get("_show_starred", False)
            st.rerun()

    # ── Export ───────────────────────────────────────────────────────
    if st.session_state.messages:
        st.divider()
        st.download_button(
            label="⬇️ Export chat (.md)",
            data=export_chat_md(),
            file_name=f"chat_{st.session_state.user_id[:8]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # ── Session stats ────────────────────────────────────────────────
    if st.session_state.messages:
        st.divider()
        q_cnt  = sum(1 for m in st.session_state.messages if m["role"] == "user")
        fb_pos = sum(1 for v in st.session_state.feedback.values() if v == 1)
        fb_neg = sum(1 for v in st.session_state.feedback.values() if v == -1)
        c1, c2 = st.columns(2)
        c1.metric("Queries", q_cnt)
        c2.metric("Docs",    doc_count)
        if fb_pos or fb_neg:
            st.caption(f"Feedback: 👍 {fb_pos}  👎 {fb_neg}")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='margin-bottom:2px'>💬 RAG AI Chat</h1>"
    "<p style='color:var(--text-mute);margin-top:0;margin-bottom:16px;font-size:.95rem'>"
    "Ask questions about your documents — answers stream in real time</p>",
    unsafe_allow_html=True,
)

# ── Mode indicator ───────────────────────────────────────────────────────
mode_color = "var(--accent2)" if st.session_state.doc_mode else "var(--warn)"
mode_text  = "📄 Document mode — answers grounded in your uploads" \
             if st.session_state.doc_mode else \
             "💬 General mode — answering without document context"
st.markdown(
    f"<div style='font-size:.8rem;color:{mode_color};margin-bottom:12px'>{mode_text}</div>",
    unsafe_allow_html=True,
)

# ── Starred panel ────────────────────────────────────────────────────────
if st.session_state.get("_show_starred"):
    st.markdown("### ⭐ Starred Answers")
    starred_hist = get_query_history(starred_only=True)
    if starred_hist:
        for q in starred_hist:
            st.markdown(
                f"""<div class="history-card">
                    <div class="h-q">⭐ Q: {q['question']}</div>
                    <div class="h-a">{q['answer'][:400]}…</div>
                    <div class="h-meta">🕐 {q['created_at'][:16].replace('T',' ')}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No starred answers yet. Click ☆ on any answer to pin it.")
    st.divider()

# ── History panel ────────────────────────────────────────────────────────
if st.session_state.show_history:
    st.markdown("### 📜 Interaction History")
    history = get_query_history()
    if history:
        for group_label, items in _group_history(history).items():
            st.markdown(
                f"<div style='color:var(--text-mute);font-size:.72rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:.07em;margin:16px 0 8px'>"
                f"{group_label}</div>",
                unsafe_allow_html=True,
            )
            for q in items:
                ans  = q["answer"][:300].replace("\n", " ")
                ts   = q["created_at"][:16].replace("T", " ")
                fb   = " 👍" if q.get("feedback") == 1 else " 👎" if q.get("feedback") == -1 else ""
                star = " ⭐" if q.get("starred") else ""
                mode = " 📄" if q.get("doc_mode") else " 💬"
                st.markdown(
                    f"""<div class="history-card">
                        <div class="h-q">Q: {q['question']}{fb}{star}{mode}</div>
                        <div class="h-a">{ans}…</div>
                        <div class="h-meta">🕐 {ts} · 📚 {q['chunk_count']} chunks</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
        st.divider()
    else:
        st.info("No history yet.")

# ── Messages ─────────────────────────────────────────────────────────────
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

# ── Input row ────────────────────────────────────────────────────────────
in_col, chat_col = st.columns([1, 20])
with in_col:
    st.markdown('<div class="plus-btn">', unsafe_allow_html=True)
    if st.button("＋", key="open_upload", help="Upload a document"):
        upload_modal()
    st.markdown("</div>", unsafe_allow_html=True)
with chat_col:
    prefill    = st.session_state.pop("_prefill", None)
    user_input = st.chat_input("Ask me anything about your documents…")
    if prefill and not user_input:
        user_input = prefill

# ── Handle submission ────────────────────────────────────────────────────
if user_input:
    st.session_state.suggestions = []
    ts_now = datetime.now().strftime("%H:%M")

    with st.chat_message("user"):
        st.markdown(f'<div class="msg-ts">{ts_now}</div>', unsafe_allow_html=True)
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input, "ts": ts_now})

    with st.chat_message("assistant"):
        status_box  = st.empty()
        answer_box  = st.empty()
        full_answer = ""
        sources_ok  = False

        status_box.markdown(
            "<div class='status-badge'>🔍 Searching documents…</div>",
            unsafe_allow_html=True,
        )

        try:
            for token in stream_question(user_input, top_k=top_k,
                                         temperature=temperature, max_tokens=max_tokens):
                if not sources_ok:
                    sources_ok = True
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

        chunks = st.session_state.pop("_pending_chunks", [])
        qid    = st.session_state.pop("_pending_query_id", None)
        sim    = st.session_state.pop("_pending_similarity", 0.0)
        usage  = st.session_state.pop("_pending_token_usage", None)

        if usage:
            st.markdown(
                f"<div style='font-size:0.75rem;color:var(--text-mute);margin-bottom:8px;'>"
                f"⚡ Tokens: <b>{usage.get('total_tokens', 0)}</b> "
                f"(Prompt: {usage.get('prompt_tokens', 0)} | "
                f"Completion: {usage.get('completion_tokens', 0)})</div>",
                unsafe_allow_html=True,
            )

        # Copy button — safe: content stored in hidden span, not inlined in JS
        import html as _html
        live_copy_id  = f"copy_live_{qid or 'latest'}"
        safe_answer   = _html.escape(full_answer)
        st.markdown(
            f"""<span id="{live_copy_id}" style="display:none">{safe_answer}</span>
            <button class="copy-btn" data-copy-id="{live_copy_id}"
                onclick="copyFromElement(this)">📋 Copy</button>""",
            unsafe_allow_html=True,
        )

        render_sources(chunks)
        st.markdown(f'<div class="msg-ts">{ts_now}</div>', unsafe_allow_html=True)

        # Feedback + star
        if qid:
            c1, c2, c3, _ = st.columns([1, 1, 1, 9])
            with c1:
                if st.button("👍", key=f"up_new_{qid}", use_container_width=True):
                    st.session_state.feedback[qid] = 1
                    submit_feedback(qid, 1)
                    st.toast("Thanks!", icon="👍")
            with c2:
                if st.button("👎", key=f"dn_new_{qid}", use_container_width=True):
                    st.session_state.feedback[qid] = -1
                    submit_feedback(qid, -1)
                    st.toast("Noted.", icon="👎")
            with c3:
                if st.button("☆ Pin", key=f"star_new_{qid}", use_container_width=True):
                    st.session_state.starred.add(qid)
                    submit_star(qid, True)
                    st.toast("Pinned!", icon="⭐")

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  full_answer,
        "chunks":   chunks,
        "similarity": sim,
        "query_id": qid,
        "ts":       ts_now,
        "token_usage": usage,
    })

    # Follow-up suggestions
    if full_answer and not full_answer.startswith("⚠️"):
        sugs = get_suggestions(user_input, full_answer)
        if sugs:
            st.session_state.suggestions = sugs
            st.rerun()
