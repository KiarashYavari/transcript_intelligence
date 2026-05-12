"""
embedding_pipeline.py
---------------------

Embedding pipeline for RAG documents.

Uses SentenceTransformers to generate local embeddings.
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from processed.src.rag.config import EMBEDDING_MODEL_NAME
from processed.src.rag.document_builder import RAGDocument


logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Generates embeddings for RAG documents.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Loaded embedding model: %s", model_name)

        except Exception as error:
            logger.exception("Failed to load embedding model: %s", model_name)
            raise error

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Generate embeddings for text inputs.
        """

        if not texts:
            return []

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

            return embeddings.tolist()

        except Exception as error:
            logger.exception("Failed to embed texts")
            raise error

    def embed_documents(
        self,
        documents: list[RAGDocument],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Generate embeddings for RAG documents.
        """

        texts = [document.text for document in documents]

        return self.embed_texts(
            texts=texts,
            batch_size=batch_size,
        )