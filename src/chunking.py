from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split on sentence-ending punctuation followed by space or newline.
        # Using re.split with a capturing group keeps the delimiter attached
        # to the preceding sentence (via the recombine step below).
        parts = re.split(r'([.!?]\s+|[.]\n)', text)

        # re.split with a capturing group produces alternating tokens:
        # ["sentence", ". ", "sentence", "! ", "sentence", ""]
        # Zip pairs back together so each sentence keeps its punctuation.
        sentences: list[str] = []
        for i in range(0, len(parts) - 1, 2):
            sentence = (parts[i] + parts[i + 1]).strip()
            if sentence:
                sentences.append(sentence)
        # Catch any trailing text that had no terminating punctuation
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        # Group sentences into chunks of max_sentences_per_chunk
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group))

        return chunks if chunks else [text]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: text fits within chunk_size — no further splitting needed
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Fallback: no separators left — return as-is even if oversized
        if not remaining_separators:
            return [current_text]

        separator = remaining_separators[0]
        rest = remaining_separators[1:]

        # Split on current separator (empty string splits into individual chars)
        if separator == "":
            pieces = list(current_text)
        else:
            pieces = current_text.split(separator)

        # Recurse on any piece that is still too large, using next separators
        small_pieces: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                small_pieces.append(piece)
            else:
                small_pieces.extend(self._split(piece, rest))

        # Merge consecutive small pieces into larger chunks (greedy packing)
        # This prevents ending up with one tiny piece per chunk slot.
        merged: list[str] = []
        current_chunk = ""
        for piece in small_pieces:
            candidate = (current_chunk + separator + piece).lstrip(separator) if current_chunk else piece
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = piece
        if current_chunk:
            merged.append(current_chunk)

        return merged if merged else [current_text]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size).chunk(text),
            "by_sentences": SentenceChunker().chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }

        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count else 0.0
            result[name] = {
                "count": count,
                "avg_length": round(avg_length, 2),
                "chunks": chunks,
            }
        return result
