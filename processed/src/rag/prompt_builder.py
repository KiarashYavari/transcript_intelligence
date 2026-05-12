"""
prompt_builder.py
-----------------

Builds prompts from retrieved transcript context.
"""

from __future__ import annotations

from processed.src.rag.retriever import RetrievedDocument


class PromptBuilder:
    """
    Converts retrieved documents into an LLM-ready prompt.
    """

    @staticmethod
    def build_context(
        retrieved_documents: list[RetrievedDocument],
    ) -> str:
        """
        Build readable context block from retrieved documents.
        """

        context_blocks: list[str] = []

        for index, document in enumerate(retrieved_documents, start=1):
            metadata = document.metadata

            block = (
                f"[Source {index}]\n"
                f"Document Type: {metadata.get('document_type', 'unknown')}\n"
                f"Meeting ID: {metadata.get('meeting_id', 'unknown')}\n"
                f"Call Type: {metadata.get('call_type', 'unknown')}\n"
                f"Speaker: {metadata.get('speaker_name', 'unknown')}\n"
                f"Sentiment: {metadata.get('sentiment', 'unknown')}\n"
                f"Topics: {metadata.get('topics', '')}\n"
                f"Text:\n{document.text}\n"
            )

            context_blocks.append(block)

        return "\n" + ("-" * 80 + "\n").join(context_blocks)

    @staticmethod
    def build_prompt(
        question: str,
        retrieved_documents: list[RetrievedDocument],
    ) -> str:
        """
        Build final RAG prompt.
        """

        context = PromptBuilder.build_context(retrieved_documents)

        return f"""
You are a Transcript Intelligence assistant for a B2B SaaS company.

Use only the provided context to answer the question.
If the context is insufficient, say so clearly.

Your answer should include:
1. Direct answer
2. Supporting evidence from the retrieved context
3. Business implication for stakeholders

Context:
{context}

Question:
{question}

Answer:
""".strip()