from typing import Any, Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def _retrieve(
        self,
        question: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if metadata_filter:
            return self._store.search_with_filter(
                question,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        return self._store.search(question, top_k=top_k)

    def _format_context(self, results: list[dict[str, Any]]) -> str:
        context_blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            title = metadata.get("doc_title") or metadata.get("source") or metadata.get("doc_id") or result.get("id")
            source = metadata.get("source", "unknown")
            chunk_index = metadata.get("chunk_index", "?")
            score = result.get("score", 0.0)
            content = result.get("content", "").strip()
            context_blocks.append(
                f"[Source {index}]\n"
                f"title: {title}\n"
                f"source: {source}\n"
                f"chunk_index: {chunk_index}\n"
                f"retrieval_score: {score:.4f}\n"
                f"content:\n{content}"
            )
        return "\n\n---\n\n".join(context_blocks)

    def _format_sources(self, results: list[dict[str, Any]]) -> str:
        lines = ["\n\nSources:"]
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            title = metadata.get("doc_title") or metadata.get("source") or metadata.get("doc_id") or result.get("id")
            source = metadata.get("source", "unknown")
            chunk_index = metadata.get("chunk_index", "?")
            score = result.get("score", 0.0)
            lines.append(f"- [{index}] {title} | {source} | chunk {chunk_index} | score={score:.3f}")
        return "\n".join(lines)

    def build_prompt(
        self,
        question: str,
        results: list[dict[str, Any]],
    ) -> str:
        context = self._format_context(results)
        return (
            "You are a retrieval-augmented knowledge base assistant.\n"
            "Answer the user's question using only the provided context sources.\n"
            "If the context does not contain enough information, say that the documents do not provide enough evidence.\n"
            "Do not invent facts, dates, scores, procedures, or policy details.\n"
            "When possible, cite sources inline using [Source 1], [Source 2], etc.\n"
            "Answer in the same language as the user's question unless the user asks otherwise.\n\n"
            f"Context sources:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    def answer(
        self,
        question: str,
        top_k: int = 3,
        metadata_filter: dict[str, Any] | None = None,
        include_sources: bool = True,
    ) -> str:
        results = self._retrieve(question, top_k=top_k, metadata_filter=metadata_filter)
        if not results:
            return "I could not find relevant context in the knowledge base."

        prompt = self.build_prompt(question, results)
        response = self._llm_fn(prompt).strip()
        if include_sources:
            response += self._format_sources(results)
        return response

    def answer_with_sources(
        self,
        question: str,
        top_k: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        results = self._retrieve(question, top_k=top_k, metadata_filter=metadata_filter)
        if not results:
            return {"answer": "I could not find relevant context in the knowledge base.", "sources": []}

        prompt = self.build_prompt(question, results)
        return {
            "answer": self._llm_fn(prompt).strip(),
            "sources": [
                {
                    "id": result.get("id"),
                    "score": result.get("score"),
                    "metadata": result.get("metadata", {}),
                    "content": result.get("content", ""),
                }
                for result in results
            ],
        }
