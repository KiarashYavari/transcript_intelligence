"""
rag_runner.py
-------------

Command-line runner for building and querying the RAG system.

Usage:

Build vector database:
    python -m processed.src.rag.rag_runner --build

Ask a question:
    python -m processed.src.rag.rag_runner --query "What are customers unhappy about?"

Ask with filters:
    python -m processed.src.rag.rag_runner --query "What are the major issues?" --call-type customer_support
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from processed.src.rag.document_builder import RAGDocumentBuilder
from processed.src.rag.embedding_pipeline import EmbeddingPipeline
from processed.src.rag.query_engine import RAGQueryEngine
from processed.src.rag.vector_store import VectorStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def build_vector_database(reset: bool = True) -> None:
    """
    Build ChromaDB vector database from parquet tables.
    """

    builder = RAGDocumentBuilder()
    documents = builder.build_all_documents()

    if not documents:
        raise ValueError("No RAG documents were generated")

    embedding_pipeline = EmbeddingPipeline()
    embeddings = embedding_pipeline.embed_documents(documents)

    vector_store = VectorStore()

    if reset:
        vector_store.reset_collection()

    vector_store.add_documents(
        documents=documents,
        embeddings=embeddings,
    )

    print(f"Vector database built successfully.")
    print(f"Documents stored: {vector_store.count()}")


def run_query(
    question: str,
    top_k: int,
    filters: dict[str, Any] | None,
) -> None:
    """
    Run a RAG query and print retrieved context plus prompt.
    """

    engine = RAGQueryEngine()

    response = engine.query(
        question=question,
        top_k=top_k,
        filters=filters,
    )

    print("\n" + "=" * 90)
    print("RETRIEVED DOCUMENTS")
    print("=" * 90)

    for index, document in enumerate(response.retrieved_documents, start=1):
        print(f"\nSOURCE {index}")
        print("-" * 90)
        print(f"Distance: {document.distance}")
        print(f"Metadata: {document.metadata}")
        print(document.text[:1000])

    print("\n" + "=" * 90)
    print("RAG PROMPT")
    print("=" * 90)
    print(response.prompt)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Transcript Intelligence RAG Runner"
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="Build or rebuild vector database",
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Question to ask the RAG system",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve",
    )

    parser.add_argument(
        "--call-type",
        type=str,
        default=None,
        help="Optional metadata filter for call_type",
    )

    parser.add_argument(
        "--sentiment",
        type=str,
        default=None,
        help="Optional metadata filter for sentiment",
    )

    parser.add_argument(
        "--document-type",
        type=str,
        default=None,
        help="Optional metadata filter for document_type",
    )

    return parser.parse_args()


def build_filters(args: argparse.Namespace) -> dict[str, Any] | None:
    """
    Build Chroma metadata filters from CLI arguments.
    """

    filters: dict[str, Any] = {}

    if args.call_type:
        filters["call_type"] = args.call_type

    if args.sentiment:
        filters["sentiment"] = args.sentiment

    if args.document_type:
        filters["document_type"] = args.document_type

    return filters or None


def main() -> None:
    """
    Main CLI entrypoint.
    """

    args = parse_args()

    if args.build:
        build_vector_database(reset=True)

    if args.query:
        filters = build_filters(args)

        run_query(
            question=args.query,
            top_k=args.top_k,
            filters=filters,
        )

    if not args.build and not args.query:
        print("Nothing to do. Use --build or --query.")


if __name__ == "__main__":
    main()