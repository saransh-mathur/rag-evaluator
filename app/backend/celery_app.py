import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from celery import Celery
from dotenv import load_dotenv

# Load workspace environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "rag_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)
