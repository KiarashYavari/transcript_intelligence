"""
query_engine.py
---------------

High-level RAG query engine.

This module retrieves relevant context and builds an answer-ready prompt.
The first version returns the prompt and retrieved documents, so you can plug in
any LLM provider later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from processed.src.rag.prompt_builder import PromptBuilder
from processed.src.rag.retriever import RetrievedDocument, Retriever


logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    Response from the RAG query engine.
    """

    question: str
    prompt: str
    retrieved_documents: list[RetrievedDocument]


class RAGQueryEngine:
    """
    Coordinates retrieval and prompt construction.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        prompt_builder: PromptBuilder | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def query(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """
        Retrieve relevant documents and build a final prompt.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        retrieved_documents = self.retriever.search(
            query=question,
            top_k=top_k,
            filters=filters,
        )

        prompt = self.prompt_builder.build_prompt(
            question=question,
            retrieved_documents=retrieved_documents,
        )

        return RAGResponse(
            question=question,
            prompt=prompt,
            retrieved_documents=retrieved_documents,
        )