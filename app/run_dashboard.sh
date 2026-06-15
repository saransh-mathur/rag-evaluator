#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 streamlit run src/dashboard.py -- --run-dir runs/latest
