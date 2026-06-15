"""Migration v3: add new columns to documents and query_history."""

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import sqlalchemy as sa

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/rag_ai")
engine = sa.create_engine(DATABASE_URL)

MIGRATIONS = [
    "ALTER TABLE documents    ADD COLUMN IF NOT EXISTS summary        TEXT;",
    "ALTER TABLE documents    ADD COLUMN IF NOT EXISTS collection_tag VARCHAR(100);",
    "ALTER TABLE query_history ADD COLUMN IF NOT EXISTS feedback      INTEGER;",
    "ALTER TABLE query_history ADD COLUMN IF NOT EXISTS starred       BOOLEAN DEFAULT FALSE;",
    "ALTER TABLE query_history ADD COLUMN IF NOT EXISTS doc_mode      BOOLEAN DEFAULT TRUE;",
]

if __name__ == "__main__":
    with engine.connect() as conn:
        for sql in MIGRATIONS:
            conn.execute(sa.text(sql))
        conn.commit()
    print("✅ Migration v3 complete.")
