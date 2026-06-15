#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
set -a
source .env
set +a
cd frontend
PYTHONDONTWRITEBYTECODE=1 streamlit run app.py --server.port=${STREAMLIT_PORT:-8501}
