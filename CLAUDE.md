# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository is a student-facing Python lab on embeddings, chunking, vector search, and a simple retrieval-augmented agent. The main implementation surface is the `src` package; the tests in `tests/test_solution.py` define the expected public API and behavior.

The lab is intentionally incomplete: several methods raise `NotImplementedError` and are meant to be filled in by students.

## Common commands

### Setup
```bash
pip install -r requirements.txt
```

### Run tests
```bash
pytest tests/ -v
```

### Run a single test class
```bash
pytest tests/test_solution.py::TestEmbeddingStore -v
```

### Run a single test method
```bash
pytest tests/test_solution.py::TestEmbeddingStore::test_search_returns_list -v
```

### Run the manual demo
```bash
python3 main.py
python3 main.py "What do these documents say about vector stores?"
```

### Verify optional embedding backends
Local Sentence Transformers backend:
```bash
pip install sentence-transformers
python3 - <<'PY'
from src import LocalEmbedder
embedder = LocalEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

OpenAI backend:
```bash
pip install openai
export OPENAI_API_KEY=your-key-here
python3 - <<'PY'
from pathlib import Path
from dotenv import load_dotenv
from src import OpenAIEmbedder

load_dotenv(dotenv_path=Path('.env'), override=False)
embedder = OpenAIEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

## Architecture

### Public package boundary
`src/__init__.py` is the package API. Tests import the package dynamically and expect these symbols to be exported from `src`, not only from individual modules. If you rename or move internals, preserve the public exports used by `tests/test_solution.py`.

### Core data flow
The lab models a simple RAG pipeline:

1. `Document` in `src/models.py` is the shared data container for raw text plus metadata.
2. Chunkers in `src/chunking.py` split text into retrievable units using three strategies:
   - `FixedSizeChunker`: implemented reference example.
   - `SentenceChunker`: sentence-grouped chunking.
   - `RecursiveChunker`: separator-priority chunking with recursive fallback.
3. Embedding backends in `src/embeddings.py` convert text to vectors:
   - `MockEmbedder` / `_mock_embed` is the deterministic default used by tests and default classroom runs.
   - `LocalEmbedder` and `OpenAIEmbedder` are optional runtime backends.
4. `EmbeddingStore` in `src/store.py` stores embedded documents and performs similarity search.
5. `KnowledgeBaseAgent` in `src/agent.py` retrieves top-k chunks from the store, builds a prompt, and delegates answer generation to an injected `llm_fn` callable.

### Store design
`EmbeddingStore` is designed with two execution paths:

- Default and test path: an in-memory list of normalized records.
- Optional path: ChromaDB if `chromadb` is importable.

Even though ChromaDB is mentioned, the current codebase and tests are primarily structured around the in-memory fallback. The constructor accepts `embedding_fn` so tests can inject `_mock_embed` and remain deterministic.

### Manual demo path
`main.py` is an end-to-end demo that:

- loads `.md` and `.txt` files from `data/`
- loads environment variables via `python-dotenv`
- selects the embedding backend from `EMBEDDING_PROVIDER`
- falls back to `_mock_embed` if optional providers are unavailable
- builds an `EmbeddingStore`
- runs both direct similarity search and `KnowledgeBaseAgent.answer()`

This makes `main.py` the quickest way to sanity-check behavior beyond unit tests.

## Implementation constraints inferred from tests

- The repo is test-first: `tests/test_solution.py` is the authoritative specification for expected behavior.
- The package name defaults to `src`, but tests resolve it via `LAB_SOLUTION_PACKAGE`; avoid hard-coding assumptions that break package-level imports.
- Search results are expected to be dictionaries containing at least `content`, `score`, and metadata used by filter/delete flows.
- `search_with_filter()` is expected to pre-filter by metadata, then rank the remaining candidates.
- `delete_document()` is expected to remove all chunks associated with one logical document id.
- `ChunkingStrategyComparator.compare()` is expected to return three named strategy sections: `fixed_size`, `by_sentences`, and `recursive`, each including `count`, `avg_length`, and `chunks`.

## Working conventions specific to this repo

- Prefer keeping implementations simple and aligned to the lab’s teaching goals; this repository is an educational exercise, not a production system.
- Preserve the current fallback behavior around embeddings: the lab should still run without optional providers installed.
- When changing behavior, validate against `pytest tests/ -v` first, then use `python3 main.py` if the change affects the end-to-end retrieval flow.
