"""Database connection and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/rag_ai"
)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Get database session for dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables and seed default accounts."""
    from .models import Base, UserAccount
    Base.metadata.create_all(bind=engine)
    
    # Run migration checks dynamically
    from sqlalchemy import text
    db = SessionLocal()
    try:
        db.execute(text("ALTER TABLE user_accounts ADD COLUMN IF NOT EXISTS custom_instructions TEXT;"))
        db.execute(text("ALTER TABLE query_history ADD COLUMN IF NOT EXISTS chunk_evaluations TEXT;"))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DEBUG] Alter table warnings: {e}")
        
    try:
        # Seed admin
        admin = db.query(UserAccount).filter(UserAccount.username == "admin").first()
        if not admin:
            admin_pwd_hash = hashlib.sha256("admin123".encode()).hexdigest()
            db.add(UserAccount(username="admin", password_hash=admin_pwd_hash, role="admin"))
            
        # Seed user
        user = db.query(UserAccount).filter(UserAccount.username == "user").first()
        if not user:
            user_pwd_hash = hashlib.sha256("user123".encode()).hexdigest()
            db.add(UserAccount(username="user", password_hash=user_pwd_hash, role="user"))
            
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Context manager for database operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
