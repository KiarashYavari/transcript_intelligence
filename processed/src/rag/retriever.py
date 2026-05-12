"""
retriever.py
------------

Metadata-aware semantic retriever for Transcript Intelligence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from processed.src.rag.config import DEFAULT_TOP_K
from processed.src.rag.embedding_pipeline import EmbeddingPipeline
from processed.src.rag.vector_store import VectorStore


logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """
    Retrieved document from vector search.
    """

    text: str
    metadata: dict[str, Any]
    distance: float


class Retriever:
    """
    Retrieves relevant transcript intelligence documents.
    """

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()
        self.vector_store = vector_store or VectorStore()

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve documents relevant to a user query.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query_embedding = self.embedding_pipeline.embed_texts([query])[0]

        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        results: list[RetrievedDocument] = []

        for text, metadata, distance in zip(documents, metadatas, distances):
            results.append(
                RetrievedDocument(
                    text=text,
                    metadata=metadata or {},
                    distance=distance,
                )
            )

        return results