"""One-time migration: add feedback column to query_history if missing."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import sqlalchemy as sa

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/rag_ai",
)

engine = sa.create_engine(DATABASE_URL)

ADD_COLUMN_SQL = """
ALTER TABLE query_history
ADD COLUMN IF NOT EXISTS feedback INTEGER;
"""

if __name__ == "__main__":
    with engine.connect() as conn:
        conn.execute(sa.text(ADD_COLUMN_SQL))
        conn.commit()
    print("✅ feedback column added (or already existed).")
