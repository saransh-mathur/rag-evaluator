"""Main Streamlit RAG AI chat application."""

import streamlit as st
import requests
import json
from datetime import datetime
import uuid

# Configuration
API_BASE_URL = "http://localhost:8000/api"
TIMEOUT = 60

# Page config
st.set_page_config(
    page_title="RAG AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stChatMessage {
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []


def load_documents():
    """Load user's documents from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/documents/list",
            params={"user_id": st.session_state.user_id},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        st.session_state.documents = response.json()
        return st.session_state.documents
    except Exception as e:
        st.error(f"Failed to load documents: {e}")
        return []


def upload_document(uploaded_file):
    """Upload a document to the API."""
    try:
        files = {"file": uploaded_file}
        params = {"user_id": st.session_state.user_id}
        
        response = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files=files,
            params=params,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        return result
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def delete_document(doc_id):
    """Delete a document."""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/documents/{doc_id}",
            params={"user_id": st.session_state.user_id},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        load_documents()
        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


def ask_question(question, top_k=5, temperature=0.1):
    """Send a question to the API and get answer."""
    try:
        payload = {
            "question": question,
            "user_id": st.session_state.user_id,
            "top_k": top_k,
            "temperature": temperature
        }
        
        response = requests.post(
            f"{API_BASE_URL}/queries/ask",
            json=payload,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None


def get_query_history():
    """Get query history from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/queries/history",
            params={"user_id": st.session_state.user_id, "limit": 20},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to load history: {e}")
        return []


# Main UI
st.title("🤖 RAG AI Chat Assistant")
st.caption(f"Session: {st.session_state.user_id[:8]}...")

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col2:
    # Sidebar for documents
    st.header("📁 Documents")
    
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md"],
        key="doc_uploader"
    )
    
    if uploaded_file:
        with st.spinner("Uploading..."):
            result = upload_document(uploaded_file)
            if result:
                st.success(f"✅ Uploaded: {result['filename']}\n({result['chunks_created']} chunks)")
                load_documents()
    
    st.divider()
    
    # Document list
    st.subheader("Your Documents")
    docs = load_documents()
    
    if docs:
        for doc in docs:
            col_doc, col_del = st.columns([4, 1])
            with col_doc:
                st.caption(f"📄 {doc['filename']}\n({doc['chunks']} chunks)")
            with col_del:
                if st.button("🗑️", key=f"del_{doc['id']}", help="Delete"):
                    delete_document(doc["id"])
                    st.rerun()
    else:
        st.info("No documents yet. Upload one to get started!")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    top_k = st.slider("Results to retrieve", 1, 10, 5)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.1)

with col1:
    # Chat interface
    st.header("💬 Chat")
    
    # Display messages
    chat_container = st.container(height=400, border=True)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "chunks" in message:
                    with st.expander("📚 Sources"):
                        for chunk in message["chunks"]:
                            st.caption(f"**{chunk['filename']}** (sim: {chunk['similarity']})")
                            st.text(chunk['text'][:300] + "...")
    
    # Input
    st.divider()
    
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Ask me anything...",
            placeholder="Type your question here",
            label_visibility="collapsed",
            key="user_input"
        )
    
    with col_send:
        send_button = st.button("Send", key="send_btn", use_container_width=True)
    
    # Process input
    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.spinner("🤔 Thinking..."):
            result = ask_question(user_input, top_k=top_k, temperature=temperature)
            
            if result:
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "chunks": result["retrieved_chunks"],
                    "similarity": result["top_similarity"]
                })
                
                st.rerun()
    
    # Clear chat button
    col_clear, col_history = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col_history:
        if st.button("📜 Show History", use_container_width=True):
            st.session_state.show_history = not st.session_state.get("show_history", False)

# Show history in expander
if st.session_state.get("show_history", False):
    st.divider()
    st.subheader("📜 Query History")
    
    history = get_query_history()
    if history:
        for query in history:
            with st.expander(f"Q: {query['question'][:50]}... ({query['created_at'][:10]})"):
                st.write(f"**Answer:** {query['answer'][:200]}...")
                st.caption(f"Sources: {query['chunk_count']} chunks retrieved")
    else:
        st.info("No query history yet.")
