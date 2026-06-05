# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Văn Dưỡng
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai text chunks có high cosine similarity nghĩa là các embedding vector của chúng trỏ về cùng một hướng trong không gian vector — tức là chúng chia sẻ nhiều đặc trưng ngữ nghĩa, thường xuất hiện trong những ngữ cảnh tương tự nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "The doctor examined the patient carefully."
- Sentence B: "A physician checked the sick person thoroughly."
- Tại sao tương đồng: Cả hai câu đều mô tả cùng một hành động (bác sĩ khám bệnh nhân), chỉ dùng từ đồng nghĩa — embedding model học được sự tương đương này.

**Ví dụ LOW similarity:**
- Sentence A: "The stock market crashed by 10% today."
- Sentence B: "I love eating homemade pizza on weekends."
- Tại sao khác: Hai câu hoàn toàn khác chủ đề (tài chính vs ẩm thực), không có từ hoặc khái niệm nào chung.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo góc giữa hai vector, không phụ thuộc vào độ dài (magnitude) của chúng — điều này quan trọng vì một đoạn văn dài và một câu ngắn có thể nói về cùng chủ đề nhưng có magnitude rất khác nhau. Euclidean distance bị ảnh hưởng bởi độ dài vector, nên có thể cho kết quả sai lệch khi so sánh các văn bản có độ dài không đồng đều.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
>
> `= ceil((10000 - 50) / (500 - 50))`
> `= ceil(9950 / 450)`
> `= ceil(22.11)`
> `= **23 chunks**`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> `= ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = **25 chunks**` — tăng thêm 2 chunks.
> Overlap nhiều hơn giúp bảo toàn ngữ cảnh ở ranh giới giữa các chunk: một câu quan trọng nằm ở cuối chunk N cũng xuất hiện ở đầu chunk N+1, giúp retrieval không bỏ sót thông tin bị cắt đứt giữa hai chunks.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| | FixedSizeChunker (`fixed_size`) | | | |
| | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng `re.split(r'([.!?]\s+|[.]\n)', text)` với **capturing group** để tách câu mà vẫn giữ lại dấu kết câu đính kèm theo từng câu. Output của `re.split` với capturing group là list xen kẽ `["câu", "dấu", "câu", "dấu", ...]`, nên tôi zip các cặp `(parts[i], parts[i+1])` để reconstruct từng câu đầy đủ. Sau đó group theo `max_sentences_per_chunk` bằng slice và `" ".join()`. Edge case: trailing text không có dấu kết câu được xử lý riêng bằng kiểm tra `len(parts) % 2 == 1`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy với hai base case: (1) text đã nhỏ hơn `chunk_size` → trả về ngay, (2) hết separator → trả về dù oversized (fallback). Với mỗi separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]`, tách text thành các mảnh rồi đệ quy trên những mảnh còn quá dài với `remaining_separators[1:]`. Bước quan trọng nhất là **greedy packing**: sau khi có nhiều mảnh nhỏ, gom chúng lại miễn tổng `≤ chunk_size` để tránh tạo ra hàng trăm chunks cực nhỏ. Separator `""` là fallback cuối cùng — tách thành từng ký tự.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi `Document` được chuyển thành một `dict` record gồm 4 fields: `id`, `content`, `embedding` (kết quả gọi `embedding_fn`), và `metadata` (copy từ `doc.metadata` kèm thêm `doc_id`). Tất cả records được lưu trong `self._store: list[dict]`. Khi search, embed query thành vector rồi tính **dot product** với `embedding` của từng record (vì vectors đã được normalize bởi MockEmbedder, dot product ≈ cosine similarity). Sort descending theo score và trả về top_k.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter`: Guard `if not metadata_filter` trước để xử lý `None` case — nếu không có filter thì search toàn bộ store. Nếu có filter, dùng list comprehension với `all(r["metadata"].get(k) == v for k, v in metadata_filter.items())` để lọc records match tất cả key-value, rồi gọi `_search_records` trên subset đó. `delete_document`: lưu `before = len(self._store)`, dùng list comprehension giữ lại records **không** match `doc_id`, so sánh size trước/sau để trả về `True/False` — clean pattern, không cần flag biến.

### KnowledgeBaseAgent

**`answer`** — approach:
> Implement đúng theo RAG pattern 3 bước: (1) **Retrieve** — gọi `self._store.search(question, top_k=top_k)` để lấy top-k chunks liên quan nhất; (2) **Augment** — join các chunks bằng `"\n\n"` thành `context`, format prompt theo cấu trúc `Context:\n{context}\n\nQuestion: {question}\n\nAnswer:`; (3) **Generate** — gọi `self._llm_fn(prompt)` và trả về kết quả. Agent là thin orchestrator, không chứa logic nặng.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

42 passed in 0.04s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | The cat sat on the mat. | A feline rested on the rug. | high | -0.0506 | ❌ |
| 2 | Python is a programming language. | Python is used for data science. | high | -0.0162 | ❌ |
| 3 | I love eating pizza. | The stock market crashed today. | low | -0.0517 | ✅ |
| 4 | Machine learning uses neural networks. | Deep learning is a subset of machine learning. | high | -0.1842 | ❌ |
| 5 | The sun rises in the east. | My car needs an oil change. | low | -0.0253 | ✅ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Điều bất ngờ nhất là **tất cả scores đều âm và gần bằng 0**, kể cả các cặp câu rõ ràng có nghĩa tương đồng như pair 1 (cat/feline) và pair 4 (machine learning/deep learning). Điều này xảy ra vì `MockEmbedder` tạo vector bằng hash MD5 — hoàn toàn ngẫu nhiên, không mang thông tin ngữ nghĩa. Để có similarity scores phản ánh đúng nghĩa, cần dùng embedder thật như `all-MiniLM-L6-v2` hoặc `text-embedding-3-small` — các model này được train để đưa những câu có nghĩa gần nhau vào gần nhau trong không gian vector.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
