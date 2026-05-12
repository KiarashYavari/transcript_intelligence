"""
vector_store.py
---------------

Local ChromaDB vector store for Transcript Intelligence RAG.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from processed.src.rag.config import COLLECTION_NAME, VECTOR_DB_DIR
from processed.src.rag.document_builder import RAGDocument


logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper around ChromaDB persistent vector store.
    """

    def __init__(
        self,
        persist_directory=VECTOR_DB_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )

            logger.info("Initialized Chroma collection: %s", collection_name)

        except Exception as error:
            logger.exception("Failed to initialize vector store")
            raise error

    def reset_collection(self) -> None:
        """
        Delete and recreate the collection.
        """

        try:
            self.client.delete_collection(self.collection_name)

        except Exception:
            logger.warning(
                "Collection did not exist or could not be deleted: %s",
                self.collection_name,
            )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        logger.info("Reset collection: %s", self.collection_name)

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Chroma metadata values must be primitive types.

        Converts unsupported values to strings or removes null-like values.
        """

        clean: dict[str, Any] = {}

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)

        return clean

    def add_documents(
        self,
        documents: list[RAGDocument],
        embeddings: list[list[float]],
        batch_size: int = 500,
    ) -> None:
        """
        Add documents and embeddings to ChromaDB.
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "documents and embeddings must have the same length"
            )

        if not documents:
            logger.warning("No documents to add")
            return

        try:
            for start in range(0, len(documents), batch_size):
                end = start + batch_size
                batch_documents = documents[start:end]
                batch_embeddings = embeddings[start:end]

                self.collection.upsert(
                    ids=[doc.document_id for doc in batch_documents],
                    documents=[doc.text for doc in batch_documents],
                    embeddings=batch_embeddings,
                    metadatas=[
                        self._clean_metadata(doc.metadata)
                        for doc in batch_documents
                    ],
                )

            logger.info("Added %s documents to vector store", len(documents))

        except Exception as error:
            logger.exception("Failed to add documents to vector store")
            raise error

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Query the vector store.
        """

        try:
            where_filter = filters if filters else None

            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

        except Exception as error:
            logger.exception("Vector store query failed")
            raise error

    def count(self) -> int:
        """
        Return number of documents in collection.
        """

        return self.collection.count()