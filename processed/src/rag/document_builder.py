"""
document_builder.py
-------------------

Builds retrieval-ready documents from processed parquet tables.

The main source of retrieval is transcript_chunks, enriched with metadata from:
- meetings
- topics
- meeting_summaries
- action_items
- key_moments

Usage:
    python -m processed.src.rag.document_builder
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from processed.src.rag.config import PROCESSED_DIR


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGDocument:
    """
    A single retrieval-ready document.

    Attributes:
        document_id: Unique document identifier.
        document_type: Type of document.
        text: Text to embed and retrieve.
        meeting_id: Meeting identifier.
        metadata: Additional metadata used for filtering and context.
    """

    document_id: str
    document_type: str
    text: str
    meeting_id: str
    metadata: dict[str, Any]


class RAGDocumentBuilder:
    """
    Builds RAG documents from parquet tables.
    """

    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.processed_dir = processed_dir

        self.meetings_df = self._load_table("meetings.parquet")
        self.transcript_chunks_df = self._load_table("transcript_chunks.parquet")
        self.meeting_summaries_df = self._load_table("meeting_summaries.parquet")
        self.action_items_df = self._load_table("action_items.parquet")
        self.key_moments_df = self._load_table("key_moments.parquet")
        self.topics_df = self._load_table("topics.parquet")

    def _load_table(self, file_name: str) -> pd.DataFrame:
        """
        Load a parquet table safely.
        """

        file_path = self.processed_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Parquet table not found: {file_path}")

        try:
            dataframe = pd.read_parquet(file_path)
            logger.info("Loaded table: %s", file_name)
            return dataframe

        except Exception as error:
            logger.exception("Failed to load table: %s", file_name)
            raise error

    @staticmethod
    def _safe_text(value: Any) -> str:
        """
        Convert a value to clean string text.
        """

        if pd.isna(value):
            return ""

        return str(value).strip()

    def _meeting_metadata_map(self) -> dict[str, dict[str, Any]]:
        """
        Convert meetings dataframe into meeting-level metadata lookup.
        """

        metadata_map: dict[str, dict[str, Any]] = {}

        for _, row in self.meetings_df.iterrows():
            meeting_id = self._safe_text(row.get("meeting_id"))

            if not meeting_id:
                continue

            metadata_map[meeting_id] = {
                "meeting_id": meeting_id,
                "title": self._safe_text(row.get("title")),
                "call_type": self._safe_text(row.get("call_type")),
                "duration_minutes": row.get("duration_minutes"),
                "start_time": self._safe_text(row.get("start_time")),
            }

        return metadata_map

    def _topics_by_meeting(self) -> dict[str, list[str]]:
        """
        Build a meeting_id -> list of topics lookup.
        """

        if not {"meeting_id", "topic"}.issubset(self.topics_df.columns):
            return {}

        grouped = (
            self.topics_df.dropna(subset=["meeting_id", "topic"])
            .groupby("meeting_id")["topic"]
            .apply(lambda values: sorted(set(str(v).strip() for v in values if str(v).strip())))
        )

        return grouped.to_dict()

    def build_transcript_documents(self) -> list[RAGDocument]:
        """
        Build RAG documents from transcript chunks.
        """

        required_columns = {
            "meeting_id",
            "chunk_index",
            "sentence",
        }

        missing = required_columns - set(self.transcript_chunks_df.columns)

        if missing:
            raise ValueError(
                f"transcript_chunks table is missing required columns: {missing}"
            )

        meeting_metadata = self._meeting_metadata_map()
        topics_by_meeting = self._topics_by_meeting()

        documents: list[RAGDocument] = []

        for _, row in self.transcript_chunks_df.iterrows():
            meeting_id = self._safe_text(row.get("meeting_id"))
            chunk_index = row.get("chunk_index")
            sentence = self._safe_text(row.get("sentence"))

            if not meeting_id or not sentence:
                continue

            document_id = f"transcript_chunk::{meeting_id}::{chunk_index}"

            metadata = {
                **meeting_metadata.get(meeting_id, {}),
                "document_type": "transcript_chunk",
                "meeting_id": meeting_id,
                "chunk_index": chunk_index,
                "speaker_id": self._safe_text(row.get("speaker_id")),
                "speaker_name": self._safe_text(row.get("speaker_name")),
                "sentiment": self._safe_text(row.get("sentiment")),
                "start_time_seconds": row.get("start_time_seconds"),
                "end_time_seconds": row.get("end_time_seconds"),
                "confidence_score": row.get("confidence_score"),
                "topics": ", ".join(topics_by_meeting.get(meeting_id, [])),
            }

            text = (
                f"Speaker: {metadata.get('speaker_name', '')}\n"
                f"Sentiment: {metadata.get('sentiment', '')}\n"
                f"Transcript: {sentence}"
            )

            documents.append(
                RAGDocument(
                    document_id=document_id,
                    document_type="transcript_chunk",
                    text=text,
                    meeting_id=meeting_id,
                    metadata=metadata,
                )
            )

        return documents

    def build_summary_documents(self) -> list[RAGDocument]:
        """
        Build RAG documents from meeting summaries.
        """

        if "meeting_id" not in self.meeting_summaries_df.columns:
            return []

        text_column = self._detect_text_column(
            self.meeting_summaries_df,
            ["summary", "summary_text", "meeting_summary"],
        )

        if text_column is None:
            return []

        meeting_metadata = self._meeting_metadata_map()
        documents: list[RAGDocument] = []

        for _, row in self.meeting_summaries_df.iterrows():
            meeting_id = self._safe_text(row.get("meeting_id"))
            summary = self._safe_text(row.get(text_column))

            if not meeting_id or not summary:
                continue

            metadata = {
                **meeting_metadata.get(meeting_id, {}),
                "document_type": "meeting_summary",
                "meeting_id": meeting_id,
            }

            documents.append(
                RAGDocument(
                    document_id=f"meeting_summary::{meeting_id}",
                    document_type="meeting_summary",
                    text=f"Meeting Summary:\n{summary}",
                    meeting_id=meeting_id,
                    metadata=metadata,
                )
            )

        return documents

    def build_action_item_documents(self) -> list[RAGDocument]:
        """
        Build RAG documents from action items.
        """

        if "meeting_id" not in self.action_items_df.columns:
            return []

        text_column = self._detect_text_column(
            self.action_items_df,
            ["action_item", "action", "text", "description"],
        )

        if text_column is None:
            return []

        meeting_metadata = self._meeting_metadata_map()
        documents: list[RAGDocument] = []

        for index, row in self.action_items_df.iterrows():
            meeting_id = self._safe_text(row.get("meeting_id"))
            action_text = self._safe_text(row.get(text_column))

            if not meeting_id or not action_text:
                continue

            metadata = {
                **meeting_metadata.get(meeting_id, {}),
                "document_type": "action_item",
                "meeting_id": meeting_id,
                "assignee": self._safe_text(row.get("assignee")),
                "due_date": self._safe_text(row.get("due_date")),
            }

            documents.append(
                RAGDocument(
                    document_id=f"action_item::{meeting_id}::{index}",
                    document_type="action_item",
                    text=f"Action Item:\n{action_text}",
                    meeting_id=meeting_id,
                    metadata=metadata,
                )
            )

        return documents

    def build_key_moment_documents(self) -> list[RAGDocument]:
        """
        Build RAG documents from key moments.
        """

        if "meeting_id" not in self.key_moments_df.columns:
            return []

        text_column = self._detect_text_column(
            self.key_moments_df,
            ["key_moment", "moment", "text", "description"],
        )

        if text_column is None:
            return []

        meeting_metadata = self._meeting_metadata_map()
        documents: list[RAGDocument] = []

        for index, row in self.key_moments_df.iterrows():
            meeting_id = self._safe_text(row.get("meeting_id"))
            moment_text = self._safe_text(row.get(text_column))

            if not meeting_id or not moment_text:
                continue

            metadata = {
                **meeting_metadata.get(meeting_id, {}),
                "document_type": "key_moment",
                "meeting_id": meeting_id,
            }

            documents.append(
                RAGDocument(
                    document_id=f"key_moment::{meeting_id}::{index}",
                    document_type="key_moment",
                    text=f"Key Moment:\n{moment_text}",
                    meeting_id=meeting_id,
                    metadata=metadata,
                )
            )

        return documents

    @staticmethod
    def _detect_text_column(
        dataframe: pd.DataFrame,
        candidates: list[str],
    ) -> str | None:
        """
        Detect the first available text column from candidate names.
        """

        for column in candidates:
            if column in dataframe.columns:
                return column

        return None

    def build_all_documents(self) -> list[RAGDocument]:
        """
        Build all supported RAG documents.
        """

        documents: list[RAGDocument] = []

        documents.extend(self.build_transcript_documents())
        documents.extend(self.build_summary_documents())
        documents.extend(self.build_action_item_documents())
        documents.extend(self.build_key_moment_documents())

        logger.info("Built %s RAG documents", len(documents))

        return documents


def main() -> None:
    """
    Build RAG documents and print a small preview.
    """

    logging.basicConfig(level=logging.INFO)

    builder = RAGDocumentBuilder()
    documents = builder.build_all_documents()

    print(f"Built {len(documents)} documents")

    for document in documents[:3]:
        print("-" * 80)
        print(document.document_id)
        print(document.document_type)
        print(document.text[:500])
        print(document.metadata)


if __name__ == "__main__":
    main()