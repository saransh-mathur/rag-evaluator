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

# Synchronize with browser localStorage to auto-restore session
st.components.v1.html(
    """
    <script>
    try {
        var parentUrl = new URL(window.parent.location.href);
        var sessionId = parentUrl.searchParams.get("session_id");
        if (sessionId) {
            localStorage.setItem("rag_session_id", sessionId);
        } else {
            var storedId = localStorage.getItem("rag_session_id");
            if (storedId) {
                parentUrl.searchParams.set("session_id", storedId);
                window.parent.location.href = parentUrl.toString();
            }
        }
    } catch (e) {
        console.log("Local storage sync error:", e);
    }
    </script>
    """,
    height=0,
    width=0,
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
    "strategy":      "hybrid",
    "persona":       "standard",
}

# Determine the active session ID from query params
q_session_id = st.query_params.get("session_id")

if "user_id" not in st.session_state:
    if q_session_id:
        print(f"[DEBUG] Found session_id in query parameters: {q_session_id}")
        st.session_state.user_id = q_session_id
        st.session_state["_needs_restore"] = True
    else:
        new_id = str(uuid.uuid4())
        print(f"[DEBUG] No session ID in query params. Generating new UUID: {new_id}")
        st.session_state.user_id = new_id
        st.query_params["session_id"] = new_id
        st.session_state["_needs_restore"] = False

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

def check_backend_health() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL.replace('/api', '')}/health", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "offline", "dependencies": {}}


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


def get_query_detail(query_id: int) -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/queries/history/{query_id}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[DEBUG] Failed to load query detail for {query_id}: {e}")
        return {}


def restore_session_chat(user_id: str):
    print(f"[DEBUG] Restoring chat history for user_id: {user_id}")
    st.session_state.user_id = user_id
    history = get_query_history() # fetches past database Q&As
    chat_msgs = []
    # history comes sorted desc (newest first). Let's reverse it to have chronological order (oldest first).
    for q in reversed(history):
        chat_msgs.append({
            "role": "user",
            "content": q["question"],
            "ts": q["created_at"][11:16] if len(q.get("created_at", "")) > 16 else ""
        })
        chat_msgs.append({
            "role": "assistant",
            "content": q["answer"],
            "chunks": q.get("retrieved_chunks", []),
            "similarity": q.get("top_similarity", 0.0),
            "query_id": q["id"],
            "ts": q["created_at"][11:16] if len(q.get("created_at", "")) > 16 else "",
            "token_usage": None,
            "chunk_evaluations": q.get("chunk_evaluations", {})
        })
    st.session_state.messages = chat_msgs
    
    # Also reload starred and feedback states!
    st.session_state.starred = set()
    st.session_state.feedback = {}
    for q in history:
        qid = q["id"]
        if q.get("starred"):
            st.session_state.starred.add(qid)
        if q.get("feedback") is not None:
            st.session_state.feedback[qid] = q["feedback"]


def list_users() -> list:
    try:
        r = requests.get(f"{API_BASE_URL}/queries/users", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[DEBUG] Failed to list sessions: {e}")
        return []


def check_login(username, password) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE_URL}/queries/auth/login",
            json={"username": username, "password": password},
            timeout=TIMEOUT
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def register_user(username, password) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE_URL}/queries/auth/register",
            json={"username": username, "password": password},
            timeout=TIMEOUT
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_admin_metrics() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/queries/admin/metrics", timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[DEBUG] Failed to load admin metrics: {e}")
    return {}


def clear_db_history() -> bool:
    try:
        r = requests.post(f"{API_BASE_URL}/queries/admin/clear-history", timeout=TIMEOUT)
        if r.status_code == 200:
            return True
    except Exception as e:
        print(f"[DEBUG] Failed to clear database history: {e}")
    return False


def get_users_report() -> list:
    try:
        r = requests.get(f"{API_BASE_URL}/queries/admin/users-report", timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[DEBUG] Failed to load users report: {e}")
    return []


def delete_user_account(username: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE_URL}/queries/admin/users/{username}", timeout=TIMEOUT)
        if r.status_code == 200:
            return True
    except Exception as e:
        print(f"[DEBUG] Failed to delete user: {e}")
    return False


def change_user_role(username: str, role: str) -> bool:
    try:
        r = requests.post(
            f"{API_BASE_URL}/queries/admin/users/change-role",
            json={"username": username, "role": role},
            timeout=TIMEOUT
        )
        if r.status_code == 200:
            return True
    except Exception as e:
        print(f"[DEBUG] Failed to change user role: {e}")
    return False


def clear_current_session_history(session_id: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE_URL}/queries/history/clear-session/{session_id}", timeout=TIMEOUT)
        if r.status_code == 200:
            return True
    except Exception as e:
        print(f"[DEBUG] Failed to clear session history: {e}")
    return False


def render_admin_telemetry_ui():
    col_title, col_action = st.columns([3, 1])
    with col_title:
        st.markdown("### 📊 Admin Telemetry & Metrics Console")
        st.caption("Monitoring real-time API load, database writes, and OCR performance")
    with col_action:
        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Purge DB", use_container_width=True, type="primary"):
            confirm_clear_history_modal()

    tab_global, tab_users = st.tabs(["📊 Global Telemetry", "👥 User Management"])

    with tab_global:
        metrics = get_admin_metrics()
        if not metrics:
            st.info("No metrics logged yet. Try querying or uploading documents.")
        else:
            # 1. Top row metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("TPM (Tokens/Min)", f"⚡ {metrics.get('tpm', 0)}")
            c2.metric("RPM (Reqs/Min)", f"🔄 {metrics.get('rpm', 0)}")
            
            ocr_success = metrics.get("ocr_success", 0)
            ocr_failure = metrics.get("ocr_failure", 0)
            ocr_total = ocr_success + ocr_failure
            ocr_rate = f"{(ocr_success / ocr_total * 100):.1f}%" if ocr_total > 0 else "N/A"
            c3.metric("OCR Success Rate", ocr_rate, f"👍 {ocr_success} / 👎 {ocr_failure}")
            
            embed_success = metrics.get("embed_success", 0)
            embed_failure = metrics.get("embed_failure", 0)
            embed_total = embed_success + embed_failure
            embed_rate = f"{(embed_success / embed_total * 100):.1f}%" if embed_total > 0 else "N/A"
            c4.metric("Embedding Rate", embed_rate, f"👍 {embed_success} / 👎 {embed_failure}")
            
            st.divider()
            
            # 2. Charts comparison
            st.markdown("#### 📈 Operations Summary")
            import pandas as pd
            chart_data = pd.DataFrame({
                "Success": [ocr_success, embed_success],
                "Failure": [ocr_failure, embed_failure]
            }, index=["OCR PDF/Img", "Embedding runs"])
            st.bar_chart(chart_data)
            
            st.divider()
            
            # 3. System Metrics Logs table
            st.markdown("#### 📜 Telemetry Log Traces (Last 100)")
            logs_data = metrics.get("logs", [])
            if logs_data:
                df_logs = pd.DataFrame(logs_data)
                df_logs = df_logs[["created_at", "event_type", "username", "tokens", "latency", "details"]]
                df_logs.columns = ["Timestamp", "Event Type", "User", "Tokens", "Latency (s)", "Details"]
                st.dataframe(df_logs, use_container_width=True, hide_index=True)
            else:
                st.caption("No log traces captured yet.")

    with tab_users:
        st.markdown("#### 👥 User Performance & Account Reports")
        st.caption("Detailed metrics per registered user account. Manage access, promote to admin, or completely purge accounts.")
        
        users_data = get_users_report()
        if not users_data:
            st.info("No user accounts found.")
        else:
            for u in users_data:
                uname = u["username"]
                role_icon = "👑" if u["role"] == "admin" else "👤"
                role_str = u["role"].capitalize()
                
                with st.expander(f"{role_icon} {uname} — {role_str} ({u['queries_count']} queries, {u['documents_count']} docs)", expanded=False):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(
                            f"""**Account Details:**
- **Role**: `{u['role']}`
- **Registered At**: `{u['created_at'].replace('T', ' ')[:19]}`
- **Active Chat Sessions**: `{u['sessions_count']}`
- **Total Queries Ran**: `{u['queries_count']}`
- **Documents Uploaded**: `{u['documents_count']}`
- **Total Tokens Consumed**: `{u['total_tokens']:,}`
- **Average LLM Response Latency**: `{u['avg_latency']}s`
- **Feedback Rating**: 👍 `{u['feedback_positive']}` / 👎 `{u['feedback_negative']}`"""
                        )
                    with c2:
                        st.markdown("**Account Actions:**")
                        if uname != "admin" and uname != st.session_state.username:
                            if st.button("🗑️ Purge Account", key=f"del_user_{uname}", use_container_width=True, type="primary"):
                                confirm_delete_user_modal(uname)
                                
                            if u["role"] == "user":
                                if st.button("👑 Promote to Admin", key=f"promo_{uname}", use_container_width=True):
                                    if change_user_role(uname, "admin"):
                                        st.toast(f"Promoted {uname} to Admin!", icon="👑")
                                        st.rerun()
                            else:
                                if st.button("👤 Demote to User", key=f"demote_{uname}", use_container_width=True):
                                    if change_user_role(uname, "user"):
                                        st.toast(f"Demoted {uname} to User.", icon="👤")
                                        st.rerun()
                        else:
                            st.info("Root Admin or Currently Logged-in User (Actions Disabled)")


def load_custom_instructions(username: str) -> str:
    try:
        r = requests.get(f"{API_BASE_URL}/queries/auth/custom-instructions/{username}", timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json().get("custom_instructions", "")
    except Exception as e:
        print(f"[DEBUG] Failed to load custom instructions: {e}")
    return ""


def save_custom_instructions(username: str, instructions: str) -> bool:
    try:
        r = requests.post(
            f"{API_BASE_URL}/queries/auth/custom-instructions",
            json={"username": username, "custom_instructions": instructions},
            timeout=TIMEOUT
        )
        if r.status_code == 200:
            return True
    except Exception as e:
        print(f"[DEBUG] Failed to save custom instructions: {e}")
    return False


def render_custom_instructions_ui():
    st.markdown("### ✍️ Custom Instructions")
    st.caption("Help the AI understand your profile and query preferences. These details will be dynamically injected into the LLM system prompts.")
    
    st.divider()
    
    current_instructions = load_custom_instructions(st.session_state.username)
    
    part1 = ""
    part2 = ""
    if "|||" in current_instructions:
        parts = current_instructions.split("|||", 1)
        part1 = parts[0].strip()
        part2 = parts[1].strip()
    else:
        part1 = current_instructions
        
    st.markdown("##### 👤 What would you like the AI to know about you to provide better responses?")
    p1 = st.text_area(
        "Who are you, what do you work on, or what is your typical workflow?",
        value=part1,
        placeholder="e.g. I am a software engineer studying machine learning datasets. Explain concepts with high technical detail. / I am a business analyst writing summaries for executives.",
        height=150,
        help="Provide context about your background, expertise, or the project you are working on."
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🎭 How would you like the AI to respond?")
    p2 = st.text_area(
        "Format, tone, styling, or length preferences?",
        value=part2,
        placeholder="e.g. Keep answers objective and factual. Format equations in LaTeX. Use bullet points and summary tables when appropriate.",
        height=150,
        help="Specify tone, language, detail level, formatting rules, or boundaries (e.g. 'explain simply', 'keep it under 300 words')."
    )
    
    st.divider()
    
    if st.button("💾 Save Instructions", use_container_width=True, type="primary"):
        combined = f"{p1.strip()}\n|||\n{p2.strip()}" if p2.strip() else p1.strip()
        if save_custom_instructions(st.session_state.username, combined):
            st.toast("Custom instructions saved successfully!", icon="💾")
            st.success("✅ Saved! These rules will guide the assistant's future answers.")
        else:
            st.error("Failed to save custom instructions.")


# Trigger restoration if needed on startup
if st.session_state.get("_needs_restore"):
    st.session_state["_needs_restore"] = False
    restore_session_chat(st.session_state.user_id)


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
    # Apply persona instructions to the query sent to LLM
    style_suffix = ""
    persona = st.session_state.get("persona", "standard")
    if persona == "academic":
        style_suffix = "\n\nFormat your response as an objective, academic-style analysis with complete context and rigorous citations."
    elif persona == "brief":
        style_suffix = "\n\nFormat your response as a concise executive brief highlighting key bullet points."
    elif persona == "coder":
        style_suffix = "\n\nProvide technical, precise details, including code blocks where appropriate."

    payload = {
        "question":    question + style_suffix,
        "user_id":     st.session_state.user_id,
        "top_k":       top_k,
        "strategy":    st.session_state.get("strategy", "hybrid"),
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


def render_sources(chunks: list, chunk_evaluations: dict = None):
    if not chunks:
        return
    
    is_admin = st.session_state.get("role") == "admin"
    
    with st.expander(f"📚 {len(chunks)} source chunk(s) used", expanded=False):
        for chunk in chunks:
            pct   = int(chunk["similarity"] * 100)
            color = "#4a9eff" if pct >= 70 else "#f0a500" if pct >= 50 else "#6a8aaf"
            
            # If admin, evaluate the chunk relevance
            eval_html = ""
            border_style = ""
            
            if is_admin:
                cid = str(chunk.get("chunk_id") or chunk.get("id") or "")
                eval_info = None
                if chunk_evaluations:
                    eval_info = chunk_evaluations.get(cid)
                    if not eval_info and cid.isdigit():
                        eval_info = chunk_evaluations.get(int(cid))
                
                if eval_info:
                    relevance = eval_info.get("relevance", 3)
                    reason = eval_info.get("reason", "No explanation provided.")
                    
                    if relevance >= 4:
                        eval_color = "#28a745" # Green
                        eval_text = f"🟢 Highly Useful (Score: {relevance}/5)"
                        border_style = f"border-left: 5px solid {eval_color};"
                    elif relevance >= 2:
                        eval_color = "#ffc107" # Yellow
                        eval_text = f"🟡 Partially Useful (Score: {relevance}/5)"
                        border_style = f"border-left: 5px solid {eval_color};"
                    else:
                        eval_color = "#dc3545" # Red
                        eval_text = f"🔴 Irrelevant / Low Utility (Score: {relevance}/5)"
                        border_style = f"border-left: 5px solid {eval_color};"
                        
                    eval_html = f"""
                    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--border); font-size: 0.8rem;">
                        <span style="font-weight: 600; color: {eval_color};">{eval_text}</span>
                        <div style="color: var(--text-dim); margin-top: 2px;"><b>Judge explanation:</b> {reason}</div>
                    </div>
                    """
                else:
                    eval_html = """
                    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--border); font-size: 0.8rem; color: var(--text-mute);">
                        ⏳ <i>Chunk relevance evaluation running in background... Rerun/interact to update.</i>
                    </div>
                    """
            
            st.markdown(
                f"""<div class="source-card" style="{border_style}">
                    <div class="src-header">📄 {chunk['filename']}</div>
                    <div class="src-text">{chunk['text']}</div>
                    <div class="src-bar-bg">
                      <div class="src-bar-fill" style="width:{pct}%;background:{color}"></div>
                    </div>
                    <small style="color:{color}">Similarity {pct}%</small>
                    {eval_html}
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
                evals = msg.get("chunk_evaluations")
                
                # If we are admin and evaluations are empty/missing, try fetching from the database
                if st.session_state.get("role") == "admin" and not evals and qid:
                    detail = get_query_detail(qid)
                    if detail and detail.get("chunk_evaluations"):
                        evals = detail["chunk_evaluations"]
                        msg["chunk_evaluations"] = evals  # cache it in message dict
                
                # If still not ready and we are admin, show a refresh button
                if st.session_state.get("role") == "admin" and not evals and qid:
                    if st.button("🔄 Refresh Evaluations", key=f"ref_eval_{idx}_{qid}"):
                        detail = get_query_detail(qid)
                        if detail and detail.get("chunk_evaluations"):
                            msg["chunk_evaluations"] = detail["chunk_evaluations"]
                            st.toast("Evaluations loaded!", icon="✅")
                            st.rerun()
                        else:
                            st.toast("Evaluation still running. Please wait a second and try again.", icon="⏳")

                render_sources(msg["chunks"], chunk_evaluations=evals)

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
            <p>Upload a document using <b>＋</b> in the input row, then ask questions about it.<br>
            Configure your retrieval settings and assistant persona in the <b>⚙️ Settings</b> expander.</p>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("#### 💡 Try one of these example prompts:")
    
    EXAMPLE_CARDS = [
        {
            "category": "📝 Summarization",
            "desc": "Condense and extract key takeaways.",
            "question": "Summarize the key points of this document."
        },
        {
            "category": "❓ Direct QA",
            "desc": "Find answers to specific queries.",
            "question": "What is the main topic covered?"
        },
        {
            "category": "🔍 Fact Extraction",
            "desc": "Locate specific data points and facts.",
            "question": "What are the most important facts mentioned?"
        },
        {
            "category": "📊 Quantitative Analysis",
            "desc": "Identify numbers, dates, and metrics.",
            "question": "Are there any numbers, dates, or statistics?"
        }
    ]

    cols = st.columns(2)
    for i, card in enumerate(EXAMPLE_CARDS):
        with cols[i % 2]:
            st.markdown(
                f"""<div style='background:var(--panel2); border:1px solid var(--border); border-radius:12px; padding:16px; margin-bottom:8px;'>
                    <h4 style='color:var(--accent2); margin:0 0 4px 0; font-size:1.0rem;'>{card["category"]}</h4>
                    <p style='color:var(--text-dim); font-size:0.8rem; margin:0 0 12px 0; min-height:32px;'>{card["desc"]}</p>
                </div>""",
                unsafe_allow_html=True
            )
            if st.button(f"👉 Use Prompt", key=f"ex_{i}", use_container_width=True):
                st.session_state["_prefill"] = card["question"]
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
# Database Maintenance modal
# ---------------------------------------------------------------------------
@st.dialog("⚠️ Confirm Database Purge")
def confirm_clear_history_modal():
    st.markdown(
        "<div style='margin-bottom:16px;line-height:1.6;color:var(--text-dim);'>"
        "Are you sure you want to <b>permanently delete all query histories, chat sessions, and telemetry logs</b> from the database?"
        "<br><br><span style='color:var(--error);font-weight:bold;'>⚠️ This action is irreversible.</span>"
        "</div>",
        unsafe_allow_html=True
    )
    
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("🔴 Yes, Purge Database", use_container_width=True, type="primary"):
            if clear_db_history():
                st.session_state.messages = []
                st.session_state.suggestions = []
                st.toast("Database purged successfully!", icon="🗑️")
                st.rerun()
            else:
                st.error("Failed to clear database history.")
    with col_no:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("⚠️ Confirm User Deletion")
def confirm_delete_user_modal(username: str):
    st.markdown(
        f"<div style='margin-bottom:16px;line-height:1.6;color:var(--text-dim);'>"
        f"Are you sure you want to permanently delete the user account <b>'{username}'</b>?"
        f"<br><br>This will <b>completely purge</b> all their documents, session histories, chat logs, and telemetry traces from the database."
        f"<br><br><span style='color:var(--error);font-weight:bold;'>⚠️ This action is irreversible.</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("🔴 Purge User Account", use_container_width=True, type="primary"):
            if delete_user_account(username):
                st.toast(f"User {username} deleted successfully!", icon="🗑️")
                st.rerun()
            else:
                st.error("Failed to delete user account.")
    with col_no:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ── Login Page ───────────────────────────────────────────────────────────
if not st.session_state.get("logged_in"):
    _, col_login, _ = st.columns([1, 2, 1])
    with col_login:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """<div style='text-align:center; margin-bottom:20px;'>
                <h1 style='color:var(--accent); font-size:2.2rem; letter-spacing: -1px;'>🔐 RAG AI Console</h1>
                <p style='color:var(--text-mute);'>Enter your credentials to access the workspace</p>
            </div>""",
            unsafe_allow_html=True
        )
        
        login_tab, register_tab = st.tabs(["🔑 Sign In", "📝 Create Account"])
        
        with login_tab:
            l_username = st.text_input("Username", key="login_username")
            l_password = st.text_input("Password", type="password", key="login_password")
            if st.button("Sign In", use_container_width=True, key="login_btn"):
                if not l_username or not l_password:
                    st.error("Please fill in all fields")
                else:
                    auth = check_login(l_username.strip(), l_password)
                    if auth:
                        st.session_state.logged_in = True
                        st.session_state.username = auth["username"]
                        st.session_state.role = auth["role"]
                        st.session_state.user_id = auth["username"]
                        st.query_params["session_id"] = auth["username"]
                        st.toast(f"Welcome back, {auth['username']}!", icon="👋")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                        
        with register_tab:
            r_username = st.text_input("New Username", key="reg_username")
            r_password = st.text_input("New Password", type="password", key="reg_password")
            if st.button("Sign Up", use_container_width=True, key="reg_btn"):
                if not r_username or not r_password:
                    st.error("Please fill in all fields")
                else:
                    reg = register_user(r_username.strip(), r_password)
                    if reg:
                        st.session_state.logged_in = True
                        st.session_state.username = r_username.strip()
                        st.session_state.role = "user"
                        st.session_state.user_id = r_username.strip()
                        st.query_params["session_id"] = r_username.strip()
                        st.toast(f"Welcome, {r_username.strip()}! Account created successfully.", icon="👋")
                        st.rerun()
                    else:
                        st.error("Registration failed. Username may already exist.")
                        
    st.stop()

# ── Session Resolution & Enforce Ownership ──────────────────────────────
if st.session_state.get("logged_in"):
    username = st.session_state.username
    current_q_session = st.query_params.get("session_id")
    
    # Force query param to match account session namespace username_UUID
    if not current_q_session or not current_q_session.startswith(username + "_"):
        all_sessions = list_users()
        user_sessions = [s["id"] for s in all_sessions if s["id"].startswith(username + "_")]
        
        if user_sessions:
            active_session = user_sessions[0]
            print(f"[DEBUG] Re-injecting last session for {username}: {active_session}")
        else:
            active_session = f"{username}_{uuid.uuid4()}"
            print(f"[DEBUG] Generating first session for {username}: {active_session}")
            
        st.session_state.user_id = active_session
        st.query_params["session_id"] = active_session
        st.session_state["_needs_restore"] = True

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── User Profile & Logout ──────────────────────────────────────────
    st.markdown(f"**👤 Account Info**")
    role_badge = "👑 Admin" if st.session_state.role == "admin" else "👤 User"
    st.markdown(
        f"""<div style='background:var(--panel2); padding:8px 12px; border:1px solid var(--border); border-radius:8px; margin-bottom:8px; font-size:0.8rem;'>
        <strong>User</strong>: {st.session_state.username}<br>
        <strong>Role</strong>: {role_badge}
        </div>""",
        unsafe_allow_html=True
    )
    
    # Workspace View Selector
    views = ["💬 Chat Sandbox", "✍️ Custom Instructions"]
    if st.session_state.role == "admin":
        views.append("📊 Telemetry Dashboard")
    st.radio("Workspace View", views, key="workspace_view_selector")
    st.divider()
        
    if st.button("🚪 Log Out", use_container_width=True, key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.session_state.messages = []
        st.session_state.documents = []
        st.query_params.clear()
        st.rerun()

    st.divider()

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True, key="new_chat"):
        for k, v in _defaults.items():
            st.session_state[k] = v
        # User-isolated session ID
        new_id = f"{st.session_state.username}_{uuid.uuid4()}"
        st.session_state.user_id = new_id
        st.query_params["session_id"] = new_id
        print(f"[DEBUG] Starting a new chat session: {new_id}")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 RAG AI Chat")
    if st.session_state.role == "admin":
        st.caption("Active Session ID:")
        st.code(st.session_state.user_id, language="text")
    st.divider()

    # ── Load Past Sessions ──────────────────────────────────────────
    sessions = list_users()
    if sessions:
        # Filter sessions for regular users to only show their own
        if st.session_state.role != "admin":
            username_prefix = st.session_state.username + "_"
            sessions = [s for s in sessions if s["id"].startswith(username_prefix)]
            
        session_opts = {s["id"]: f"📂 {s['id'].split('_')[-1][:8] if '_' in s['id'] else s['id'][:8]}… ({s['created_at'][5:16].replace('T', ' ')})" for s in sessions}
        current_id = st.session_state.user_id
        options_list = list(session_opts.keys())
        
        if current_id not in session_opts:
            session_opts[current_id] = f"✨ Current Session ({current_id.split('_')[-1][:8] if '_' in current_id else current_id[:8]}…)"
            options_list = [current_id] + options_list
            
        selected_session = st.selectbox(
            "📂 Switch Session",
            options=options_list,
            format_func=lambda x: session_opts.get(x, x),
            index=options_list.index(current_id) if current_id in options_list else 0,
            key="session_switch_select"
        )
        if selected_session != current_id:
            st.session_state.user_id = selected_session
            st.query_params["session_id"] = selected_session
            print(f"[DEBUG] Switching session via dropdown to: {selected_session}")
            restore_session_chat(selected_session)
            load_documents()
            st.rerun()
            
        if st.button("🧹 Clear Chat History", use_container_width=True, key="clear_sess_hist"):
            if clear_current_session_history(current_id):
                st.session_state.messages = []
                st.session_state.suggestions = []
                st.toast("Chat history cleared!", icon="🧹")
                st.rerun()
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
        if st.session_state.role == "admin":
            # Retrieval Strategy
            strategies = ["hybrid", "advanced", "basic"]
            st.session_state.strategy = st.selectbox(
                "🧠 Retrieval Strategy",
                options=strategies,
                index=strategies.index(st.session_state.get("strategy", "hybrid")),
                format_func=lambda x: {
                    "basic": "🔍 Basic (Pure Vector)",
                    "hybrid": "🔀 Hybrid (Vector + Keyword + TF-IDF Rerank)",
                    "advanced": "🚀 Advanced (Multi-Query + RRF + MMR + Cross-Encoder)"
                }.get(x, x),
                help="Choose the algorithm to fetch documents. Advanced offers semantic query expansion and diversification."
            )

            # Assistant Persona
            personas = ["standard", "academic", "brief", "coder"]
            st.session_state.persona = st.selectbox(
                "🎭 Assistant Persona",
                options=personas,
                index=personas.index(st.session_state.get("persona", "standard")),
                format_func=lambda x: {
                    "standard": "💬 Standard Assistant",
                    "academic": "🎓 Academic Researcher (Citations & Detail)",
                    "brief": "💼 Executive Brief (Key Bullet Points)",
                    "coder": "💻 Software Engineer (Technical & Precise)"
                }.get(x, x),
                help="Instructs the AI on formatting and styling the generated response."
            )

            st.divider()

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

            # Restore Session manually
            manual_session = st.text_input(
                "🔑 Restore Session ID",
                value=st.session_state.user_id,
                help="Enter a valid Session UUID to reload its history and documents.",
            )
            if manual_session and manual_session.strip() != st.session_state.user_id:
                try:
                    val_uuid = uuid.UUID(manual_session.strip())
                    st.session_state.user_id = str(val_uuid)
                    st.query_params["session_id"] = str(val_uuid)
                    print(f"[DEBUG] Restoring session manually to: {val_uuid}")
                    restore_session_chat(str(val_uuid))
                    load_documents()
                    st.rerun()
                except ValueError:
                    st.error("Invalid UUID format. Please check the session key.")

            st.divider()
        else:
            st.session_state.strategy = "hybrid"
            st.session_state.persona = "standard"
            import os
            top_k = int(os.getenv("TOP_K", 3))
            temperature = 0.1
            max_tokens = 2048
            st.session_state.use_hyde = False
            st.session_state.use_rerank = True

            st.session_state.doc_mode = st.toggle(
                "📄 Document mode",
                value=st.session_state.doc_mode,
                help="ON = answer from your documents only. OFF = general LLM mode.",
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

    st.divider()
    
    # ── System Health Checker ─────────────────────────────────────────
    st.markdown("**🖥️ System Status**")
    health = check_backend_health()
    if health.get("status") in ("healthy", "ok", "degraded"):
        status_text = "🟢 Online" if health.get("status") == "healthy" else "🟡 Degraded"
        postgres_status = "🟢 Postgres" if health.get("dependencies", {}).get("postgres") == "ok" else "🔴 Postgres"
        ollama_status = "🟢 Ollama" if health.get("dependencies", {}).get("ollama") == "ok" else "🔴 Ollama"
    else:
        status_text = "🔴 Offline"
        postgres_status = "⚪ Postgres"
        ollama_status = "⚪ Ollama"
        
    st.markdown(
        f"""<div style='font-size:0.75rem; background:var(--panel2); padding:10px; border:1px solid var(--border); border-radius:8px; line-height: 1.45;'>
        <strong>API Server</strong>: {status_text}<br>
        <strong>Database</strong>: {postgres_status} &nbsp;|&nbsp; <strong>Ollama</strong>: {ollama_status}
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Diagnostics & Stats ──────────────────────────────────────────
    st.markdown("**📊 Diagnostic Console**")
    total_chunks = sum(doc.get("chunks", 0) for doc in docs)
    q_cnt  = sum(1 for m in st.session_state.messages if m["role"] == "user")
    
    st.markdown(
        f"""<div style='display:grid; grid-template-columns: repeat(2, 1fr); gap:8px;'>
            <div style='background:var(--panel2); padding:8px; border:1px solid var(--border); border-radius:8px; text-align:center;'>
                <div style='font-size:0.65rem; color:var(--text-mute); text-transform: uppercase;'>Total Chunks</div>
                <div style='font-size:1.0rem; font-weight:bold; color:var(--accent2);'>{total_chunks}</div>
            </div>
            <div style='background:var(--panel2); padding:8px; border:1px solid var(--border); border-radius:8px; text-align:center;'>
                <div style='font-size:0.65rem; color:var(--text-mute); text-transform: uppercase;'>User Queries</div>
                <div style='font-size:1.0rem; font-weight:bold; color:var(--accent2);'>{q_cnt}</div>
            </div>
            <div style='background:var(--panel2); padding:8px; border:1px solid var(--border); border-radius:8px; text-align:center;'>
                <div style='font-size:0.65rem; color:var(--text-mute); text-transform: uppercase;'>Strategy</div>
                <div style='font-size:0.9rem; font-weight:bold; color:var(--accent2);'>{st.session_state.get('strategy', 'hybrid').capitalize()}</div>
            </div>
            <div style='background:var(--panel2); padding:8px; border:1px solid var(--border); border-radius:8px; text-align:center;'>
                <div style='font-size:0.65rem; color:var(--text-mute); text-transform: uppercase;'>Persona</div>
                <div style='font-size:0.9rem; font-weight:bold; color:var(--accent2);'>{st.session_state.get('persona', 'standard').capitalize()}</div>
            </div>
        </div>""",
        unsafe_allow_html=True
    )
    
    fb_pos = sum(1 for v in st.session_state.feedback.values() if v == 1)
    fb_neg = sum(1 for v in st.session_state.feedback.values() if v == -1)
    if fb_pos or fb_neg:
        st.caption(f"Feedback logs: 👍 {fb_pos} &nbsp; 👎 {fb_neg}")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
active_view = st.session_state.get("workspace_view_selector", "💬 Chat Sandbox")

if active_view == "📊 Telemetry Dashboard" and st.session_state.role == "admin":
    render_admin_telemetry_ui()
    st.stop()
elif active_view == "✍️ Custom Instructions":
    render_custom_instructions_ui()
    st.stop()

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

        # Check if we can fetch evaluations if admin (giving a split second for bg task)
        evals = None
        if st.session_state.get("role") == "admin" and qid:
            import time
            time.sleep(0.3)
            detail = get_query_detail(qid)
            if detail and detail.get("chunk_evaluations"):
                evals = detail["chunk_evaluations"]

        render_sources(chunks, chunk_evaluations=evals)
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
        "chunk_evaluations": evals,
    })

    # Follow-up suggestions
    if full_answer and not full_answer.startswith("⚠️"):
        sugs = get_suggestions(user_input, full_answer)
        if sugs:
            st.session_state.suggestions = sugs
            st.rerun()
