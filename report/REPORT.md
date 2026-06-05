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

**Domain:** VinUni Official Policies & Regulations

**Tại sao nhóm chọn domain này?**
> Nhóm chọn các tài liệu quy định chính thức của VinUniversity vì đây là domain có cấu trúc rõ ràng (policy, procedure, guideline), câu hỏi có thể verify trực tiếp từ văn bản, và metadata phân loại tự nhiên theo category. Đây cũng là thông tin thực tế hữu ích cho sinh viên VinUni, giúp bài benchmark có ý nghĩa thực tiễn cao.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | AI Talent Training Program Handbook (Sổ tay học viên AI) | VinUni / Vingroup (20K-AI-Handbook-ver2.0) | 18,650 | category: program, lang: vi |
| 2 | Vietnamese Language Program for International Students | VinUni GDL-CHS-005-V2.0 | 6,045 | category: language, lang: en |
| 3 | English Language Requirements for Undergraduate Admissions | VinUni GDL-REG-002-V4.1 | 13,632 | category: admissions, lang: en |
| 4 | Course Evaluation Policy | VinUni POL-AQA-001-V4.0 | 15,814 | category: academic, lang: en |
| 5 | Student Grade Appeal Procedures | VinUni PRC-AQA-002-V2.0 | 6,249 | category: academic, lang: en |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | string | `academic`, `admissions`, `language`, `program` | Cho phép filter theo loại tài liệu, giúp thu hẹp search space khi query rõ ngữ cảnh |
| `lang` | string | `en`, `vi` | Lọc theo ngôn ngữ, tránh trả về chunk tiếng Việt khi query bằng tiếng Anh |
| `doc_title` | string | "Course Evaluation Policy" | Giúp trace back kết quả về tài liệu gốc |
| `source` | string | `POL-AQA-001-V4.0` | Reference number để verify nguồn |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu VinUni:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| grade_appeal (6,249 chars) | FixedSizeChunker (`fixed_size`) | 18 | 394.4 | Trung bình — có thể cắt giữa câu |
| grade_appeal (6,249 chars) | SentenceChunker (`by_sentences`) | 12 | 516.8 | Tốt — giữ nguyên câu |
| grade_appeal (6,249 chars) | RecursiveChunker (`recursive`) | 18 | 345.8 | Tốt — ưu tiên tách theo đoạn |
| course_eval (15,814 chars) | FixedSizeChunker (`fixed_size`) | 46 | 392.7 | Trung bình |
| course_eval (15,814 chars) | SentenceChunker (`by_sentences`) | 53 | 295.0 | Tốt — nhiều chunk nhỏ hơn |
| course_eval (15,814 chars) | RecursiveChunker (`recursive`) | 46 | 342.5 | Tốt |
| eng_req (13,632 chars) | FixedSizeChunker (`fixed_size`) | 39 | 398.3 | Trung bình |
| eng_req (13,632 chars) | SentenceChunker (`by_sentences`) | 31 | 435.7 | Tốt — chunks lớn hơn |
| eng_req (13,632 chars) | RecursiveChunker (`recursive`) | 37 | 367.2 | Tốt |

### Strategy Của Tôi

**Loại:** RecursiveChunker (`chunk_size=400`)

**Mô tả cách hoạt động:**
> `RecursiveChunker` tách text theo danh sách separator ưu tiên `["\n\n", "\n", ". ", " ", ""]` — thử tách bằng separator có ngữ nghĩa cao nhất trước (đoạn văn `\n\n`), nếu mảnh vẫn quá dài thì đệ quy xuống separator tiếp theo (dòng, câu, từ, ký tự). Sau khi có các mảnh nhỏ, thuật toán **greedy packing** gom các mảnh liền kề lại miễn tổng độ dài ≤ `chunk_size`, tránh tạo ra quá nhiều chunks cực nhỏ.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Các tài liệu quy định của VinUni có cấu trúc theo đoạn và section rõ ràng (numbered lists, headers, procedure steps). `RecursiveChunker` khai thác được cấu trúc này bằng cách ưu tiên tách theo `\n\n` (ranh giới đoạn/section), giúp mỗi chunk giữ được một ý hoàn chỉnh thay vì bị cắt giữa một quy định. So với `FixedSizeChunker`, chunks ít bị cắt đứt giữa câu hơn; so với `SentenceChunker`, chunk count nhỏ hơn và avg length phù hợp hơn với policy text có câu dài.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality |
|-----------|----------|-------------|------------|-------------------|
| grade_appeal | SentenceChunker (best baseline) | 12 | 516.8 | Tốt — chunks lớn, giữ ngữ cảnh |
| grade_appeal | **RecursiveChunker (của tôi)** | **18** | **345.8** | **Tốt — tách theo đoạn, granular hơn** |
| course_eval | FixedSizeChunker (best baseline) | 46 | 392.7 | Trung bình |
| course_eval | **RecursiveChunker (của tôi)** | **46** | **342.5** | **Tốt — cùng count nhưng chunk gọn hơn** |
| eng_req | SentenceChunker (best baseline) | 31 | 435.7 | Tốt |
| eng_req | **RecursiveChunker (của tôi)** | **37** | **367.2** | **Tốt — cân bằng giữa granularity và context** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Dưỡng) | RecursiveChunker(400) | — | Tôn trọng cấu trúc đoạn, linh hoạt | Cần tune chunk_size |
| [Thành viên 2] | SentenceChunker | — | Giữ nguyên câu, dễ hiểu | Chunk có thể quá dài |
| [Thành viên 3] | FixedSizeChunker | — | Đơn giản, dễ kiểm soát size | Có thể cắt đứt câu |
| [Thành viên 4] | Custom strategy | — | Tối ưu cho domain | Phức tạp hơn | 

> *Sẽ cập nhật sau khi so sánh kết quả trong nhóm*

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Dựa trên benchmark với Gemini embedding, `RecursiveChunker(400)` cho kết quả retrieval tốt (score 0.74–0.83) trên VinUni policy docs. Tuy nhiên cần so sánh thêm với các strategy của thành viên khác trên cùng 5 benchmark queries để kết luận. Dự đoán `RecursiveChunker` sẽ tốt hơn `FixedSizeChunker` vì policy text có ranh giới đoạn rõ ràng, và tốt hơn `SentenceChunker` vì policy docs có nhiều câu dài dẫn đến chunks quá lớn.

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

Chạy 5 benchmark queries của nhóm trên implementation cá nhân với `RecursiveChunker(chunk_size=400)` + `gemini-embedding-2` + `gemini-2.5-flash` LLM.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What are the phases of the AI training program? | 3 phases: Phase 1 (3 weeks foundation/NỀN TẢNG), Phase 2 (3 weeks specialization), Phase 3 (6 weeks practical/THỰC CHIẾN) |
| 2 | What is the process to appeal a student grade? | Submit via AQA department: Informal appeal → Formal written appeal → Review → Decision (PRC-AQA-002) |
| 3 | What English language test scores are required for undergraduate admission? | IELTS Academic ≥6.0, TOEFL iBT ≥80, or equivalent tests as listed in GDL-REG-002 |
| 4 | How is course evaluation conducted at VinUni? | End-of-course survey for all degree courses, administered before final exams, confidential, covers course quality & teaching (POL-AQA-001) |
| 5 | What Vietnamese language proficiency is required for international students? | CHS students: A2-equivalent (BN program), MD program has specific milestones; per GDL-CHS-005 |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What are the phases of the AI training program? | `ai_handbook_chunk009` — GIAI ĐOẠN 1/2/3, NỀN TẢNG, 3 tuần đầu... | 0.744 | ✅ | "Phase 1: Foundation (3 weeks), Phase 2: Specialization (3 weeks), Phase 3: Practical at enterprise (6 weeks)" |
| 2 | What is the process to appeal a student grade? | `grade_appeal_chunk005` — grades affected by clerical errors or bias... | 0.828 | ✅ | "Student submits informal → formal written appeal → AQA reviews → decision within 5 working days" |
| 3 | What English language test scores are required? | `english_req_chunk012` — English language proficiency tests minimum scores table | 0.801 | ✅ | "IELTS Academic (min score varies), TOEFL iBT, Duolingo, PTE Academic listed with minimum scores" |
| 4 | How is course evaluation conducted? *(filter: academic)* | `course_eval_chunk012` — end-of-course evaluation at VinUniversity... | 0.826 | ✅ | "End-of-course evaluation for each degree course; includes feedback on course quality, teaching; administered before final exams; confidential" |
| 5 | What Vietnamese language proficiency is required? | `vietnamese_lang_chunk008` — BN Students: A2-equivalent proficiency... | 0.815 | ✅ | "BN program: A2-equivalent in all 4 skills; MD program: specific proficiency milestones at enrollment and before clinical rotations" |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

**Nhận xét:**
> Tất cả 5 queries đều retrieve đúng document và đúng chunk. Q4 dùng metadata filter `category=academic` giúp thu hẹp search space từ 175 chunks xuống 64 chunks (2 docs), kết quả vẫn chính xác với score 0.826. Embedding semantic (`gemini-embedding-2`, dim=3072) cho kết quả vượt trội so với MockEmbedder — scores 0.74–0.83 phản ánh đúng độ liên quan ngữ nghĩa.

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
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | — / 5 |
| **Tổng** | | **82+ / 100** |
