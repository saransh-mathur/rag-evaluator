# RAG AI Chat — Interactive Document Q&A System

An interactive **RAG (Retrieval-Augmented Generation)** system with document uploads, vector search, streaming responses, and a built-in evaluation pipeline. Ask questions about your documents using local AI models (Ollama) or cloud APIs like Google Gemini.

**Key Features:**
- 📤 Upload documents — PDF, plain text, and markdown
- 🔍 Vector search with PostgreSQL pgvector
- 🤖 Streaming answers via local LLMs (Ollama) or Google Gemini
- 💬 Interactive chat interface (Streamlit)
- 📚 Sources shown with similarity scores as the answer streams
- � Built-in RAG evaluation pipeline with a results dashboard
- 🔐 Flexible privacy — run 100% locally or connect to cloud APIs

---

## How It Works

### Request flow

```
1. UPLOAD DOCUMENT
   └─ Streamlit sends file to backend
   └─ Backend extracts text (PDF, txt, md)
   └─ Text is split into overlapping chunks
   └─ Each chunk is embedded → stored in pgvector

2. ASK A QUESTION
   └─ Question is embedded with the same model
   └─ pgvector finds the top-K most similar chunks
   └─ Sources are returned to the UI immediately

3. ANSWER IS STREAMED
   └─ Retrieved chunks become the LLM context
   └─ gemma4:12b streams tokens back token-by-token
   └─ Full answer + query saved to history
```

### Architecture

```
┌─────────────────────────────────────────────────────┐
│         Streamlit Web UI  (localhost:8501)          │
│  • Streaming chat interface                        │
│  • Document upload panel (PDF / txt / md)          │
│  • Query history browser                           │
│  • Settings: top_k, temperature                    │
└────────────────────┬────────────────────────────────┘
                     │ HTTP / SSE
┌────────────────────▼────────────────────────────────┐
│      FastAPI Backend  (localhost:8000)              │
│  • POST /api/documents/upload                      │
│  • POST /api/queries/ask          (blocking)       │
│  • POST /api/queries/ask-stream   (SSE streaming)  │
│  • GET  /api/queries/history                       │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼─────┐ ┌───▼────┐ ┌────▼──────────┐
   │PostgreSQL │ │ LLM    │ │  nomic-embed  │
   │+ pgvector │ │        │ │  (embeddings) │
   └───────────┘ └────────┘ └───────────────┘
```

### Key components

| Component | Purpose |
|-----------|---------|
| **nomic-embed-text** | Converts text → 768-dim vectors |
| **PostgreSQL + pgvector** | Stores and searches vectors |
| **deepseek-r1:7b** | Generates answers from context |
| **FastAPI** | Backend REST + SSE API |
| **Streamlit** | Web UI for chat and uploads |

---

## Prerequisites

- **Python 3.10+**
- **PostgreSQL 14+** with the pgvector extension
- **Ollama** — https://ollama.com

---

## Setup (one-time)

### 1. PostgreSQL

```bash
sudo systemctl start postgresql
sudo -u postgres psql
```

```sql
CREATE DATABASE rag_ai;
\c rag_ai
CREATE EXTENSION vector;
\q
```

Verify:
```bash
sudo -u postgres psql -d rag_ai -c "SELECT 1 FROM pg_extension WHERE extname='vector';"
```

### 2. Ollama models

```bash
# Keep this running in a dedicated terminal
ollama serve

# Pull the models (one-time, ~5 GB total)
ollama pull nomic-embed-text   # ~200 MB — embedding model
ollama pull deepseek-r1:7b     # ~4.7 GB — language model
```

### 3. Python environment

```bash
cd rag-evaluator/app
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configuration

Copy `.env.example` to `.env` and update the database password if needed:

```bash
cp .env.example .env
nano .env
```

Key values:

```bash
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/rag_ai
GEN_MODEL=deepseek-r1:7b        # swap for any Ollama model
EMBED_MODEL=nomic-embed-text
TOP_K=5
CHUNK_SIZE=700
CHUNK_OVERLAP=120
```

---

## Running

You need **3 terminals** open simultaneously.

**Terminal 1 — Ollama** (skip if already running):
```bash
ollama serve
```

**Terminal 2 — Backend API:**
```bash
cd rag-evaluator/app
source .venv/bin/activate
./run_backend.sh
# Ready when you see: Uvicorn running on http://0.0.0.0:8000
```

**Terminal 3 — Frontend:**
```bash
cd rag-evaluator/app
source .venv/bin/activate
./run_frontend.sh
# Ready when you see: Local URL: http://localhost:8501
```

Open **http://localhost:8501** in your browser.

---

## Using the app

1. Upload a document (PDF, `.txt`, or `.md`) in the right panel
2. Type a question in the chat — the answer streams token-by-token
3. Expand **Sources** under any answer to see which chunks were retrieved
4. Use **Show History** to browse past queries

---

## Evaluation pipeline

The evaluation pipeline runs test questions through the live backend and measures retrieval and answer quality against expected phrases.

**Before running:** upload your test documents via the UI or API, then note the `user_id` from the session (shown in the top caption of the UI).

```bash
source .venv/bin/activate
./run_eval.sh --user-id YOUR_USER_ID
```

Results are written to `runs/latest/`:

| File | Contents |
|------|----------|
| `summary.json` | Overall hit rates and config |
| `results.csv` | Per-question scores |
| `retrieval_hits.csv` | Per-chunk retrieval detail |
| `question_type_summary.csv` | Breakdown by question category |
| `hallucination_examples.json` | Questions where expected phrases were missing |

View results in the dashboard:
```bash
./run_dashboard.sh
# Opens at http://localhost:8502
```

---

## Project structure

```
app/
├── backend/                    # FastAPI REST + SSE API
│   ├── main.py
│   ├── api/
│   │   ├── documents.py        # Upload, list, delete endpoints
│   │   └── queries.py          # /ask, /ask-stream, /history endpoints
│   ├── db/
│   │   ├── models.py           # SQLAlchemy schema
│   │   └── connection.py       # PostgreSQL session management
│   └── services/
│       ├── chunking.py         # Text splitting (uses shared/chunking.py)
│       ├── embeddings.py       # Ollama nomic-embed-text
│       ├── retrieval.py        # pgvector similarity search
│       └── generation.py       # Ollama LLM — blocking + streaming
├── frontend/
│   └── app.py                  # Streamlit chat UI with streaming
├── src/
│   ├── evaluate_rag.py         # Evaluation runner (calls live backend)
│   ├── dashboard.py            # Streamlit results dashboard
│   ├── ingest.py               # Chunk builder for eval
│   ├── config.py               # Env-based config
│   └── utils.py                # Scoring helpers
├── shared/
│   └── chunking.py             # Canonical chunking logic (used by both trees)
├── data/
│   ├── sample_docs/            # Sample markdown documents
│   └── test_cases.json         # Evaluation questions + expected phrases
├── requirements.txt
├── .env.example
├── run_backend.sh
├── run_frontend.sh
├── run_eval.sh
└── run_dashboard.sh
```

---

## API reference

Full interactive docs at **http://localhost:8000/docs**

**Upload a document:**
```bash
curl -X POST "http://localhost:8000/api/documents/upload?user_id=USER_ID" \
  -F "file=@document.pdf"
```

**Ask a question (blocking):**
```bash
curl -X POST "http://localhost:8000/api/queries/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "user_id": "USER_ID", "top_k": 5}'
```

**Ask a question (streaming SSE):**
```bash
curl -N -X POST "http://localhost:8000/api/queries/ask-stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "user_id": "USER_ID", "top_k": 5}'
```

---

## Configuration reference

```bash
# Ollama
EMBED_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
GEN_BASE_URL=http://localhost:11434/v1
GEN_MODEL=deepseek-r1:7b
GEN_API_KEY=ollama

# PostgreSQL
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_ai

# RAG parameters
TOP_K=5           # chunks retrieved per query
CHUNK_SIZE=700    # characters per chunk
CHUNK_OVERLAP=120 # overlap between chunks

# Ports
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
```

---

## Performance tuning

| Parameter | Faster responses | More thorough answers |
|-----------|------------------|-----------------------|
| `TOP_K` | 3 | 7 |
| `CHUNK_SIZE` | 1000 | 500 |
| `CHUNK_OVERLAP` | 50 | 150 |
| `temperature` | 0.1 | 0.3 |

---

## Troubleshooting

**PostgreSQL connection refused**
```bash
sudo systemctl start postgresql
```

**Ollama not reachable**
```bash
ollama serve          # start the server
ollama list           # verify models are pulled
```

**pgvector extension missing**
```bash
sudo -u postgres psql -d rag_ai -c "CREATE EXTENSION vector;"
```

**Port already in use**
```bash
lsof -ti:8000 | xargs kill -9   # backend
lsof -ti:8501 | xargs kill -9   # frontend
```

---

## Known limitations

- **No authentication** — `user_id` is a browser-generated UUID with no server-side validation. Anyone who knows your UUID can access your documents. Intended for local, single-user use only.
- **No multi-turn context** — each question is answered independently. Reference earlier context explicitly in your question if needed.
- **No DOCX support** — PDF, plain text, and markdown are supported. DOCX files need to be exported to PDF or text first.

---

## Development

```bash
# Formatting
black backend/ frontend/ src/

# Linting
flake8 backend/ frontend/ src/
```

---

## What's implemented vs planned

✅ Document upload — PDF, txt, md  
✅ Text chunking with configurable size and overlap  
✅ Vector embeddings via nomic-embed-text  
✅ pgvector similarity search  
✅ Streaming LLM responses (SSE)  
✅ Query history  
✅ RAG evaluation pipeline  
✅ Results dashboard  

🔲 User authentication  
🔲 Multi-turn conversation context  
🔲 DOCX file support  
🔲 Docker / docker-compose setup  
🔲 Structured logging and observability  

---

## References

- [pgvector](https://github.com/pgvector/pgvector)
- [Ollama](https://ollama.com)
- [FastAPI](https://fastapi.tiangolo.com)
- [Streamlit](https://streamlit.io)
- [Nomic Embeddings](https://www.nomic.ai)

---

## License

MIT
