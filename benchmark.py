"""
VinUni Regulations — Benchmark Script
Phase 2: Run 5 benchmark queries with RecursiveChunker strategy (Duong's personal strategy).

Usage:
    python3 benchmark.py                          # mock embedder (default)
    EMBEDDING_PROVIDER=local python3 benchmark.py # sentence-transformers
    EMBEDDING_PROVIDER=gemini python3 benchmark.py # Gemini from .env
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import RecursiveChunker
from src.embeddings import LocalEmbedder, GeminiEmbedder, _mock_embed, EMBEDDING_PROVIDER_ENV
from src.models import Document
from src.store import EmbeddingStore

load_dotenv(override=False)


def _get_embedder():
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "gemini":
        try:
            emb = GeminiEmbedder()   # reads GEMINI_API_KEY + GEMINI_EMBEDDING_MODEL from env
            print(f"Embedding backend: {emb._backend_name}")
            return emb
        except Exception as e:
            print(f"GeminiEmbedder failed ({e}), falling back to mock")
    elif provider == "local":
        try:
            emb = LocalEmbedder()
            print(f"Embedding backend: {emb._backend_name}")
            return emb
        except Exception as e:
            print(f"LocalEmbedder failed ({e}), falling back to mock")
    print("Embedding backend: mock embedder (random vectors)")
    return _mock_embed


def _get_llm_fn():
    """Return a Gemini-backed LLM function if GEMINI_API_KEY is set, else demo."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key and api_key != "your-gemini-api-key-here":
        try:
            from google import genai
            model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
            client = genai.Client(api_key=api_key)

            def gemini_llm(prompt: str) -> str:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text

            print(f"LLM backend: gemini/{model}")
            return gemini_llm
        except Exception as e:
            print(f"Gemini LLM failed ({e}), using demo LLM")

    print("LLM backend: demo (mock)")
    def demo_llm(prompt: str) -> str:
        return f"[DEMO] Context preview: {prompt[200:400].replace(chr(10), ' ')}..."
    return demo_llm

# ---------------------------------------------------------------------------
# 1. Document registry — file path + metadata
# ---------------------------------------------------------------------------
VINUNI_DOCS = [
    {
        "path": "data/20K-AI-Handbook-ver2.0-da-nen.txt",
        "id": "ai_handbook",
        "metadata": {
            "doc_title": "AI Talent Training Program Handbook",
            "category": "program",
            "lang": "vi",
            "source": "VinUni / Vingroup",
        },
    },
    {
        "path": "data/GDL-CHS-005-V2.0_Vietnamese-Language-Program-for-International-Students_4.5.2026.txt",
        "id": "vietnamese_language_program",
        "metadata": {
            "doc_title": "Vietnamese Language Program for International Students",
            "category": "language",
            "lang": "en",
            "source": "GDL-CHS-005-V2.0",
        },
    },
    {
        "path": "data/GDL-REG-002-V4.1_ENGLISH-LANGUAGE-REQUIREMENTS-FOR-UNDERGRADUATE-ADMISSIONS_4.5.2026.txt",
        "id": "english_language_requirements",
        "metadata": {
            "doc_title": "English Language Requirements for Undergraduate Admissions",
            "category": "admissions",
            "lang": "en",
            "source": "GDL-REG-002-V4.1",
        },
    },
    {
        "path": "data/POL-AQA-001-V4.0_Course-Evaluation-Policy_26.12.2025.txt",
        "id": "course_evaluation_policy",
        "metadata": {
            "doc_title": "Course Evaluation Policy",
            "category": "academic",
            "lang": "en",
            "source": "POL-AQA-001-V4.0",
        },
    },
    {
        "path": "data/PRC-AQA-002_Student-Grade-Appeal-Procedures_21.01.2026.txt",
        "id": "grade_appeal_procedures",
        "metadata": {
            "doc_title": "Student Grade Appeal Procedures",
            "category": "academic",
            "lang": "en",
            "source": "PRC-AQA-002-V2.0",
        },
    },
]

# ---------------------------------------------------------------------------
# 2. Benchmark queries (nhóm thống nhất — 5 queries)
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "query": "What are the phases of the AI training program?",
        "gold_answer": "The program has 3 phases: Phase 1 (3 weeks foundation), Phase 2 (3 weeks), Phase 3 (6 weeks practical)",
        "filter": None,
    },
    {
        "id": "Q2",
        "query": "What is the process to appeal a student grade?",
        "gold_answer": "Students must submit a grade appeal via the AQA department procedure (PRC-AQA-002)",
        "filter": None,
    },
    {
        "id": "Q3",
        "query": "What English language test scores are required for undergraduate admission?",
        "gold_answer": "See English Language Requirements guideline (GDL-REG-002) for accepted tests and minimum scores",
        "filter": None,
    },
    {
        "id": "Q4",
        "query": "How is course evaluation conducted at VinUni?",
        "gold_answer": "Course evaluation is governed by POL-AQA-001 which outlines survey requirements and access matrix",
        "filter": {"category": "academic"},   # ← metadata filter query
    },
    {
        "id": "Q5",
        "query": "What Vietnamese language proficiency is required for international students?",
        "gold_answer": "International students at CHS must achieve sufficient Vietnamese proficiency for clinical clerkships",
        "filter": None,
    },
]

# ---------------------------------------------------------------------------
# 3. Load + chunk + index
# ---------------------------------------------------------------------------
def load_and_chunk(doc_cfg: dict, chunker: RecursiveChunker) -> list[Document]:
    """Load one document, chunk it, return list of chunk Documents."""
    path = Path(doc_cfg["path"])
    if not path.exists():
        print(f"  [MISSING] {path}")
        return []

    full_text = path.read_text(encoding="utf-8")
    chunks = chunker.chunk(full_text)

    docs = []
    for i, chunk_text in enumerate(chunks):
        docs.append(Document(
            id=f"{doc_cfg['id']}_chunk{i:03d}",
            content=chunk_text,
            metadata={**doc_cfg["metadata"], "doc_id": doc_cfg["id"], "chunk_index": i},
        ))
    return docs


def build_store(chunk_size: int = 400) -> tuple[EmbeddingStore, int]:
    import time
    chunker = RecursiveChunker(chunk_size=chunk_size)
    embedder = _get_embedder()
    store = EmbeddingStore(collection_name="vinuni_benchmark", embedding_fn=embedder)

    total_chunks = 0
    print(f"\n{'='*60}")
    print(f"Strategy: RecursiveChunker(chunk_size={chunk_size})")
    print(f"{'='*60}")
    for doc_cfg in VINUNI_DOCS:
        chunks = load_and_chunk(doc_cfg, chunker)
        if chunks:
            # Add one chunk at a time with small delay to avoid rate limiting
            for chunk in chunks:
                store.add_documents([chunk])
                time.sleep(0.3)
            total_chunks += len(chunks)
            print(f"  [ok] {doc_cfg['id']:35s} → {len(chunks):3d} chunks")

    print(f"\nTotal chunks indexed: {total_chunks}")
    return store, total_chunks


# ---------------------------------------------------------------------------
# 4. Run benchmark
# ---------------------------------------------------------------------------
def run_benchmark(store: EmbeddingStore, llm_fn=None) -> None:
    from src.agent import KnowledgeBaseAgent
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn) if llm_fn else None

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")

    hits = 0
    for q in BENCHMARK_QUERIES:
        print(f"\n{q['id']}: {q['query']}")
        print(f"  Gold: {q['gold_answer'][:80]}...")

        if q["filter"]:
            results = store.search_with_filter(q["query"], top_k=3, metadata_filter=q["filter"])
            print(f"  Filter: {q['filter']}")
        else:
            results = store.search(q["query"], top_k=3)

        print(f"  Top-3 retrieved chunks:")
        relevant = False
        for i, r in enumerate(results, 1):
            doc_id = r["metadata"].get("doc_id", "?")
            score = r["score"]
            preview = r["content"][:100].replace("\n", " ")
            print(f"    {i}. [{doc_id}] score={score:.4f}")
            print(f"       {preview}...")
            if score > 0.3:
                relevant = True

        if relevant:
            hits += 1
            print(f"  ✅ Relevant chunk found in top-3 (score > 0.3)")
        else:
            print(f"  ⚠️  Low scores — may not be relevant")

        if agent:
            print(f"  Agent answer:")
            answer = agent.answer(q["query"], top_k=3)
            print(f"    {answer[:200]}...")

    print(f"\n{'='*60}")
    print(f"Summary: {hits}/5 queries had relevant chunks (score > 0.3)")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store, _ = build_store(chunk_size=400)
    llm_fn = _get_llm_fn()
    run_benchmark(store, llm_fn=llm_fn)
