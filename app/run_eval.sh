#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
set -a
source .env
set +a
python src/evaluate_rag.py \
  --docs-dir data/sample_docs \
  --tests-file data/test_cases.json \
  --output-dir runs/latest \
  --top-k ${TOP_K:-4} \
  --chunk-size ${CHUNK_SIZE:-700} \
  --chunk-overlap ${CHUNK_OVERLAP:-120}
