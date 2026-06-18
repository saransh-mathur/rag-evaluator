# RAG AI Chat - Interactive Document Q&A System

An interactive **RAG (Retrieval-Augmented Generation)** system with user document uploads, vector search, and LLM responses. Ask questions about your documents using local AI models or cloud APIs like Google Gemini!

**Key Features:**
- 📤 Upload documents (markdown, text files)
- 🔍 Vector search with PostgreSQL pgvector
- 🤖 Generate answers using local LLMs (Ollama) or Google Gemini
- 💬 Interactive chat interface (Streamlit)
- 📚 See retrieved sources with similarity scores
- 🔐 Flexible privacy — run 100% locally or connect to cloud APIs

## How It Works

### The Complete Flow

```
1. USER UPLOADS DOCUMENT
   ↓
   [Streamlit Frontend]
   
2. DOCUMENT IS PROCESSED
   ↓
   [Backend splits into chunks]
   [Each chunk is embedded with nomic-embed-text]
   [Embeddings stored in PostgreSQL pgvector]
   
3. USER ASKS A QUESTION
   ↓
   [Question is embedded with same model]
   [Vector search finds most similar chunks]
   [Retrieved chunks become context]
   
4. LLM GENERATES ANSWER
   ↓
   [gemma4:12b reads context + question]
   [Generates answer using only that context]
   [Answer prevents hallucinations]
   
5. RESULT SHOWN TO USER
   ↓
   [Answer displayed in chat]
   [Source chunks shown with similarity scores]
   [Saved to query history]
```

### System Architecture

```
┌─────────────────────────────────────────────────────┐
│         Streamlit Web UI (localhost:8501)           │
│  • Chat interface                                   │
│  • Document upload panel                           │
│  • Query history browser                           │
│  • Settings (top_k, temperature)                   │
└────────────────────┬────────────────────────────────┘
                     │ (HTTP)
┌────────────────────▼────────────────────────────────┐
│      FastAPI Backend (localhost:8000)               │
│  • Document management endpoints                   │
│  • Query/QA endpoints                              │
│  • History tracking                                │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼──┐  ┌─────▼──┐  ┌─────▼──────┐
   │PostgreSQL│  │ Ollama  │  │Nomic Embed │
   │+ pgvector│  │(LLM)    │  │(Embeddings)│
   │(Vectors) │  │         │  │            │
   └─────────┘  └─────────┘  └────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **nomic-embed-text** | Converts text → 768-dimensional vectors |
| **PostgreSQL pgvector** | Stores and searches vectors efficiently |
| **gemma4:12b / Gemini** | Generates answers based on context |
| **FastAPI** | Backend API handling documents & queries |
| **Streamlit** | Web UI for chat and uploads |

## Prerequisites

Before starting, make sure you have:
- **Python 3.10+**
- **PostgreSQL 12+**
- **Ollama** (download from https://ollama.com)

---

## Setup Instructions (One-Time)

### Step 1: PostgreSQL Database Setup ⚙️

PostgreSQL stores your documents and embeddings. We need to create a database and enable the pgvector extension.

```bash
# Start PostgreSQL service
sudo systemctl start postgresql

# Open PostgreSQL console
sudo -u postgres psql
```

In the `postgres=#` prompt, run these **one command at a time**:

```sql
-- Create the database
CREATE DATABASE rag_ai;

-- Connect to it
\c rag_ai

-- Enable vector support
CREATE EXTENSION vector;

-- Exit
\q
```

**Verify it worked:**
```bash
sudo -u postgres psql -d rag_ai -c "SELECT 1 FROM pg_extension WHERE extname='vector';"
```

You should see `1` in the output. ✅

---

### Step 2: Ollama Setup (Local Models) 🤖

Ollama provides the embedding and language models. Download from https://ollama.com and install.

**Terminal 1 - Keep Ollama running:**
```bash
ollama serve
```

**Terminal 2 - Pull the models (keep running):**
```bash
# Download embedding model (~200MB)
ollama pull nomic-embed-text

# Download LLM (~8GB - takes time)
ollama pull gemma4:12b

# Verify models are downloaded
ollama list
```

You should see both models listed. **Keep `ollama serve` running!** ✅

---

### Step 3: Python Environment Setup 🐍

Now set up the application's Python dependencies.

```bash
cd /home/saranh/projects/rag-evaluator/app

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

Wait for this to complete (may take 2-3 minutes). ✅

---

### Step 4: Configuration 🔧

Check your database password and update `.env`:

```bash
# View current .env
cat .env

# Edit if needed (default user is 'postgres')
nano .env
```

If PostgreSQL user is `postgres` with no special password, the default is fine:
```
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_ai
```

If you set a password during PostgreSQL setup, update it:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/rag_ai
```

Save and exit (`Ctrl+X` → `Y` → `Enter`). ✅

---

## Running the Application 🚀

You need **3 terminals** running simultaneously:

### Terminal 1: Ollama (if not already running)
```bash
ollama serve
```

**Keep this running!** It serves the AI models.

---

### Terminal 2: Backend API

```bash
cd /home/saranh/projects/rag-evaluator/app

# Activate environment
source .venv/bin/activate

# Run backend
./run_backend.sh
```

Wait until you see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **Backend is ready!**

You can test it at:
- **API Docs:** http://localhost:8000/docs (try endpoints here)
- **ReDoc:** http://localhost:8000/redoc

---

### Terminal 3: Frontend UI

```bash
cd /home/saranh/projects/rag-evaluator/app

# Activate environment
source .venv/bin/activate

# Run frontend
./run_frontend.sh
```

Wait until you see:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

✅ **Frontend is ready!**

---

## Using the Application 💬

1. **Open** http://localhost:8501 in your browser
2. **Upload a document** (markdown or text file) in the sidebar
3. **Ask a question** in the chat
4. **View sources** with similarity scores
5. **Check history** of all your queries

### Example Workflow

**Upload:** `python_asyncio.md` containing asyncio documentation

**Ask:** "What should I use instead of time.sleep in async code?"

**Get:** "You should use `asyncio.sleep()` instead..." with sources shown

---

## Stopping the Application 🛑

Press `Ctrl+C` in each terminal to stop:
1. Frontend (Streamlit)
2. Backend (FastAPI)
3. Ollama (optional, can leave running)

---

## Project Structure

```
app/
├── backend/                    # FastAPI REST API
│   ├── main.py                # Application entry point
│   ├── api/
│   │   ├── documents.py       # File upload & management
│   │   └── queries.py         # Chat & Q&A endpoints
│   ├── db/
│   │   ├── models.py          # Database schema (SQLAlchemy)
│   │   └── connection.py      # PostgreSQL connection
│   └── services/
│       ├── embeddings.py      # Nomic embed-text integration
│       ├── retrieval.py       # Vector similarity search
│       └── generation.py      # LLM answer generation
├── frontend/                   # Streamlit Web UI
│   └── app.py                 # Chat interface
├── requirements.txt           # Python dependencies
├── .env                       # Configuration
├── run_backend.sh             # Start backend script
├── run_frontend.sh            # Start frontend script
├── setup.py                   # Automated setup
└── SETUP.md                   # Detailed setup guide
```

---

## Configuration Reference

The `.env` file controls all settings:

```bash
# ========== Ollama (Local AI) ==========
EMBED_BASE_URL=http://localhost:11434      # Ollama server address
EMBED_MODEL=nomic-embed-text               # Embedding model

# Generation (Local LLM)
GEN_BASE_URL=http://localhost:11434/v1     # LLM server address
GEN_MODEL=gemma4:12b                       # LLM model
GEN_API_KEY=ollama                         # LLM API key

# Generation (Cloud Gemini API)
# GEN_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
# GEN_MODEL=gemini-3.5-flash
# GEN_API_KEY=your_gemini_api_key

# ========== PostgreSQL Database ==========
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_ai

# ========== RAG Parameters ==========
TOP_K=5            # Number of chunks to retrieve
CHUNK_SIZE=700     # Character count per chunk
CHUNK_OVERLAP=120  # Overlap between chunks

# ========== Server Ports ==========
API_HOST=0.0.0.0
API_PORT=8000      # Backend API port
STREAMLIT_PORT=8501 # Frontend UI port
```

---

## API Reference

The backend API can be tested at **http://localhost:8000/docs**

**Document Upload:**
```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@document.txt" \
  -F "user_id=user123"
```

**Ask a Question:**
```bash
curl -X POST "http://localhost:8000/api/queries/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "user_id": "user123",
    "top_k": 5,
    "temperature": 0.1
  }'
```

Full API documentation with all endpoints available at http://localhost:8000/docs

---

## Troubleshooting

### ❌ PostgreSQL Error: "Connection refused"
```bash
sudo systemctl start postgresql
```

### ❌ Ollama Error: "Could not connect"
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Verify models
ollama list
```

### ❌ pgvector Extension Error
```bash
sudo -u postgres psql -d rag_ai -c "CREATE EXTENSION vector;"
```

### ❌ Python Import Error
```bash
cd /home/saranh/projects/rag-evaluator/app
source .venv/bin/activate
```

### ❌ Port Already in Use
```bash
lsof -ti:8000 | xargs kill -9
```

---

## Performance Tuning

Adjust `.env` for different scenarios:

| Parameter | Fast Responses | Detailed Answers |
|-----------|---|---|
| `TOP_K` | 3 | 7 |
| `CHUNK_SIZE` | 1000 | 500 |
| `CHUNK_OVERLAP` | 50 | 150 |
| `temperature` | 0.1 | 0.3 |

---

## Common Questions

**Q: Can I upload PDF files?**  
A: Currently text and markdown only. Convert PDFs to text first.

**Q: How do I keep conversation context?**  
A: Share relevant details in follow-up questions, or upload documents with context.

**Q: Why is the first request slow?**  
A: Ollama models are loaded on-demand. Subsequent requests are faster.

**Q: Can I use different LLMs?**  
A: Yes! Download any Ollama model and update `GEN_MODEL` in `.env`

---

## Development

```bash
# Code formatting
black backend/ frontend/

# Linting
flake8 backend/ frontend/

# Run tests
pytest tests/
```

---

## Future Enhancements

- [ ] User authentication & multi-user support
- [ ] Persistent conversation history
- [ ] Document summarization
- [ ] PDF/DOCX file support
- [ ] Docker containerization
- [ ] React web UI
- [ ] RAG evaluation metrics
- [ ] Streaming responses
- [ ] Model fine-tuning

---

## Status

✅ **What Works:**
- Document upload & embedding
- Vector search & retrieval
- Answer generation
- Query history
- Interactive chat interface

⚠️ **Known Limitations:**
- No user authentication
- Single-user sessions
- Text/markdown files only
- No conversation context retention

---

## License

MIT

---

## References

- [PostgreSQL pgvector](https://github.com/pgvector/pgvector)
- [Ollama](https://ollama.com)
- [FastAPI](https://fastapi.tiangolo.com)
- [Streamlit](https://streamlit.io)
- [Nomic Embeddings](https://www.nomic.ai/)
