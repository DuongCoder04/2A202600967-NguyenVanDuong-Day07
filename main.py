from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

SAMPLE_FILES = [
    "data/20K-AI-Handbook-ver2.0-da-nen.txt",
    "data/GDL-CHS-005-V2.0_Vietnamese-Language-Program-for-International-Students_4.5.2026.txt",
    "data/GDL-REG-002-V4.1_ENGLISH-LANGUAGE-REQUIREMENTS-FOR-UNDERGRADUATE-ADMISSIONS_4.5.2026.txt",
    "data/POL-AQA-001-V4.0_Course-Evaluation-Policy_26.12.2025.txt",
    "data/PRC-AQA-002_Student-Grade-Appeal-Procedures_21.01.2026.txt",
]

DEFAULT_CHUNK_SIZE = 800


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def get_llm_fn():
    """Return a Gemini-backed LLM function when GEMINI_API_KEY is configured."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key and api_key != "your-gemini-api-key-here":
        try:
            from google import genai

            model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
            client = genai.Client(api_key=api_key)

            def gemini_llm(prompt: str) -> str:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text or ""

            gemini_llm._backend_name = f"gemini/{model}"  # type: ignore[attr-defined]
            return gemini_llm
        except Exception as exc:
            print(f"Gemini LLM unavailable ({exc}); using demo LLM.")

    demo_llm._backend_name = "demo LLM"  # type: ignore[attr-defined]
    return demo_llm


def chunk_documents(documents: list[Document], chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[Document]:
    """Split loaded files into retrievable chunks while preserving metadata."""
    chunker = RecursiveChunker(chunk_size=chunk_size)
    chunks: list[Document] = []
    for doc in documents:
        for index, content in enumerate(chunker.chunk(doc.content)):
            chunks.append(
                Document(
                    id=f"{doc.id}_chunk{index:03d}",
                    content=content,
                    metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": index},
                )
            )
    return chunks


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "gemini":
        try:
            embedder = GeminiEmbedder()
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    chunks = chunk_documents(docs)
    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(chunks)

    print(f"\nStored {store.get_collection_size()} chunks in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    llm_fn = get_llm_fn()
    print(f"LLM backend: {getattr(llm_fn, '_backend_name', llm_fn.__class__.__name__)}")
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
