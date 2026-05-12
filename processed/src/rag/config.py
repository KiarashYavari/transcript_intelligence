"""
config.py
---------

Configuration values for the RAG pipeline.
"""

from __future__ import annotations

from pathlib import Path

from processed.src.pipeline.constants import PROCESSED_DIR


RAG_DIR: Path = Path("data/rag")
VECTOR_DB_DIR: Path = RAG_DIR / "chroma_db"

COLLECTION_NAME: str = "transcript_intelligence"

EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K: int = 5

SUPPORTED_DOCUMENT_TYPES: set[str] = {
    "transcript_chunk",
    "meeting_summary",
    "action_item",
    "key_moment",
}