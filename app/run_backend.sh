#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
set -a
source .env
set +a
cd backend
PYTHONDONTWRITEBYTECODE=1 python -m uvicorn main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000} --reload
