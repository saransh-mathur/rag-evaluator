# RAG AI Chat - Setup Guide

Interactive RAG system with document upload, vector search, and local LLM responses.

## Architecture

```
Frontend (Streamlit)  
    ↓
Backend API (FastAPI)  
    ↓
┌─────────────────────────────────┐
│  PostgreSQL + pgvector          │
│  (Vector storage & retrieval)   │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Ollama (Local)                 │
│  - nomic-embed-text (embeddings)│
│  - deepseek-r1:7b (LLM)         │
└─────────────────────────────────┘
```

## Prerequisites

1. **Python 3.10+**
2. **PostgreSQL 12+**
3. **Ollama** with models:
   - `nomic-embed-text` (for embeddings)
   - `deepseek-r1:7b` (for generation)

## Step 1: PostgreSQL Setup

### Install PostgreSQL (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### Create database with pgvector
```bash
# Connect to PostgreSQL
sudo -u postgres psql

# In psql:
CREATE DATABASE rag_ai;
\c rag_ai
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

### Verify (optional)
```bash
psql -U postgres -d rag_ai -c "SELECT 1 FROM pg_extension WHERE extname='vector';"
```

## Step 2: Ollama Setup

### Install Ollama
Download from https://ollama.com

### Pull models
```bash
# In a terminal, run Ollama
ollama serve

# In another terminal, pull models
ollama pull nomic-embed-text
ollama pull deepseek-r1:7b
```

Keep Ollama running while using the application!

## Step 3: Python Environment Setup

```bash
cd /home/saranh/projects/rag-evaluator/app

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 4: Configure Environment

Edit `.env` file:

```bash
nano .env
```

Update PostgreSQL password if different:

```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/rag_ai
```

## Step 5: Run the Application

You'll need **3 terminals**:

### Terminal 1: Ollama (if not already running)
```bash
ollama serve
```

### Terminal 2: Backend API
```bash
cd /home/saranh/projects/rag-evaluator/app
source .venv/bin/activate
./run_backend.sh
```

This starts FastAPI at http://localhost:8000

API documentation available at:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

### Terminal 3: Frontend
```bash
cd /home/saranh/projects/rag-evaluator/app
source .venv/bin/activate
./run_frontend.sh
```

This starts Streamlit at http://localhost:8501

## Usage

1. **Open** http://localhost:8501 in browser
2. **Upload** markdown/text files
3. **Ask** questions about your documents
4. **View** sources and similarity scores

## API Endpoints

### Documents
- `POST /api/documents/upload` - Upload document
- `GET /api/documents/list` - List user documents
- `DELETE /api/documents/{id}` - Delete document

### Queries
- `POST /api/queries/ask` - Ask a question
- `GET /api/queries/history` - Get query history
- `GET /api/queries/history/{id}` - Get query details
- `DELETE /api/queries/{id}` - Delete from history

## Troubleshooting

### PostgreSQL connection error
```
Error: could not connect to server: Connection refused
```

Make sure PostgreSQL is running:
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
```

### Ollama connection error
```
Error: Could not connect to Ollama at http://localhost:11434
```

Make sure Ollama is running:
```bash
# Terminal 1
ollama serve
```

### pgvector extension error
```
Error: pgvector extension not found
```

Create the extension:
```bash
psql -U postgres -d rag_ai -c "CREATE EXTENSION vector;"
```

### Import errors
Make sure you're in the correct directory and virtual environment:
```bash
cd /home/saranh/projects/rag-evaluator/app
source .venv/bin/activate
```

## Development

### Database migrations (future)
```bash
alembic init migrations
alembic revision --autogenerate -m "Add initial tables"
alembic upgrade head
```

### Run tests
```bash
pytest tests/
```

### Format code
```bash
black backend/ frontend/
flake8 backend/ frontend/
```

## Performance Tips

1. **Increase chunk_size** if retrieving too much context
2. **Decrease chunk_overlap** for faster processing
3. **Adjust top_k** to retrieve fewer/more relevant chunks
4. **Use higher temperature** for creative answers (0.1 = precise)

## Architecture Details

### Database Schema

**Users** - Session management
- id (PK)
- created_at

**Documents** - Uploaded files
- id (PK)
- user_id (FK)
- filename
- original_content
- uploaded_at

**DocumentChunks** - Text chunks with embeddings
- id (PK)
- document_id (FK)
- text
- embedding (Vector, 768 dims)
- chunk_index

**QueryHistory** - Chat history
- id (PK)
- user_id (FK)
- question
- answer
- retrieved_chunks_ids (JSON)
- top_similarity
- created_at

### Process Flow

1. **Upload Document**
   - Read file
   - Split into chunks
   - Embed with nomic-embed-text
   - Store in PostgreSQL

2. **Ask Question**
   - Embed question
   - Vector search (cosine similarity)
   - Retrieve top-k chunks
   - Generate answer with deepseek-r1:7b
   - Save to history

## Next Steps

- Add authentication/multi-user
- Implement conversation history
- Add document summarization
- Support more file types (PDF, docx)
- Deploy with Docker
- Add web UI (React)
- Implement RAG evaluation metrics
