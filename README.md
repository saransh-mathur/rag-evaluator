# Local RAG Evaluation

Evaluate a local retrieval-augmented generation pipeline using Ollama:

- **Embeddings:** `nomic-embed-text`
- **Generation:** `deepseek-r1:7b`
- **Dashboard:** Streamlit

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) with models pulled:

```bash
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

## Quick start

```bash
cd ~/projects/rag-eval-project/app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
./run_eval.sh
./run_dashboard.sh
```

## Project layout

| Path | Purpose |
|------|---------|
| `src/ingest.py` | Load and chunk markdown docs |
| `src/evaluate_rag.py` | Embed, retrieve, generate, score |
| `src/dashboard.py` | Streamlit results viewer |
| `data/sample_docs/` | Sample technical markdown |
| `data/test_cases.json` | Benchmark questions |
| `runs/latest/` | Current evaluation outputs |

## Tuning

Edit `.env` to change `TOP_K`, `CHUNK_SIZE`, and `CHUNK_OVERLAP`, then re-run `./run_eval.sh`.

## Outputs

After evaluation, `runs/latest/` contains:

- `summary.json` — aggregate metrics
- `results.csv` — per-question scores
- `retrieval_hits.csv` — retrieved chunks per question
- `question_type_summary.csv` — breakdown by question type
- `hallucination_examples.json` — missed answers with confident responses
