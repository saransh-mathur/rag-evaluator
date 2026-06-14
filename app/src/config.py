"""Load configuration from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent.parent
load_dotenv(APP_DIR / ".env")


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GEN_BASE_URL = os.getenv("GEN_BASE_URL", "http://localhost:11434/v1")
GEN_MODEL = os.getenv("GEN_MODEL", "deepseek-r1:7b")
GEN_API_KEY = os.getenv("GEN_API_KEY", "ollama")
TOP_K = _int("TOP_K", 4)
CHUNK_SIZE = _int("CHUNK_SIZE", 700)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 120)
