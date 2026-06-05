# Demo Guide: RAG Knowledge Base cho tài liệu VinUni

## 1. Mô tả dự án

Dự án này là một hệ thống hỏi đáp tài liệu dùng RAG (Retrieval-Augmented Generation).
Thay vì để LLM trả lời dựa trên kiến thức chung, hệ thống sẽ tìm thông tin liên quan trong bộ tài liệu trước, sau đó đưa phần context đó vào prompt để Gemini 2.5 Flash trả lời.

Bộ tài liệu nhóm dùng là các tài liệu VinUni và AI training program, gồm handbook, policy, procedure, guideline. Mục tiêu là khi người dùng hỏi về chương trình đào tạo, quy trình appeal điểm, yêu cầu tiếng Anh, course evaluation hoặc Vietnamese language requirement, hệ thống có thể tìm đúng chunk liên quan và trả lời có nguồn.

## 2. Những thành phần sử dụng

- `RecursiveChunker`: chia tài liệu dài thành các chunk nhỏ.
- `gemini-embedding-2`: biến query và chunk thành vector embedding.
- `EmbeddingStore`: vector store in-memory, lưu chunk, embedding và metadata.
- `KnowledgeBaseAgent`: agent RAG, lấy top-k chunks, build prompt và gọi LLM.
- `gemini-2.5-flash`: model sinh câu trả lời cuối cùng.
- Metadata: `source`, `doc_id`, `chunk_index`, `category`, `lang`, `doc_title`.

## 3. Tại sao dùng các thành phần này?

Em dùng `RecursiveChunker` vì tài liệu policy thường có cấu trúc đoạn, heading, numbered list. Nếu dùng fixed-size chunking thì dễ cắt giữa câu hoặc giữa ý. Recursive chunking ưu tiên tách theo đoạn trước, rồi mới xuống dòng, câu, từ, nên giữ context tốt hơn.

Em dùng embedding thay vì keyword search vì nhiều câu hỏi có thể không trùng từ khóa hoàn toàn với tài liệu. Embedding giúp tìm theo ngữ nghĩa.

Em dùng metadata vì nó giúp truy vết nguồn và có thể filter theo loại tài liệu, ví dụ `category=academic`.

Em dùng Gemini 2.5 Flash vì model nhanh, đủ tốt cho demo hỏi đáp, và trong `.env` đã cấu hình sẵn.

## 4. Flow hệ thống

1. Load 5 tài liệu `.txt` trong thư mục `data`.
2. Chia từng tài liệu thành chunks bằng `RecursiveChunker`.
3. Embed từng chunk bằng `gemini-embedding-2`.
4. Khi người dùng hỏi, embed câu hỏi và tìm top-3 chunks liên quan nhất.
5. Đưa top-3 chunks vào prompt cho `gemini-2.5-flash` trả lời, kèm source.

## 5. File cần show khi demo

- `README.md`: giới thiệu lab và mục tiêu.
- `data/`: bộ tài liệu nguồn.
- `src/chunking.py`: show `RecursiveChunker`.
- `src/store.py`: show `EmbeddingStore`, `search`, `search_with_filter`.
- `src/embeddings.py`: show `GeminiEmbedder`.
- `src/agent.py`: show prompt RAG và source citation.
- `main.py`: entry point để chạy demo.
- `benchmark.py`: 5 benchmark queries.
- `report/REPORT.md`: kết quả và phân tích.

Không nên show trực tiếp `.env` vì có API key. Nếu cần nói, chỉ nói:

```text
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_LLM_MODEL=gemini-2.5-flash
```

## 6. Câu hỏi test nên demo

```powershell
python main.py "What are the phases of the AI training program?"
```

```powershell
python main.py "What is the process to appeal a student grade?"
```

```powershell
python main.py "What English language test scores are required for undergraduate admission?"
```

```powershell
python main.py "How is course evaluation conducted at VinUni?"
```

```powershell
python main.py "What Vietnamese language proficiency is required for international students?"
```

Câu nên demo đầu tiên:

```powershell
python main.py "What are the phases of the AI training program?"
```

Lý do: output rõ ràng, top chunks đều từ AI Handbook, score cao, câu trả lời có 3 phase và có citation.

## 7. Log cần nhấn mạnh

```text
Embedding backend: gemini/gemini-embedding-2
```

Dòng này chứng minh hệ thống dùng embedding thật.

```text
Stored 94 chunks in EmbeddingStore
```

Dòng này cho thấy tài liệu đã được chunk và index.

```text
1. score=0.742 source=data\20K-AI-Handbook-ver2.0-da-nen.txt
```

Dòng này cho thấy retrieval tìm đúng tài liệu và score cao.

```text
LLM backend: gemini/gemini-2.5-flash
```

Dòng này chứng minh phần generation dùng Gemini 2.5 Flash.

```text
Sources:
- [1] ... chunk 6 | score=0.742
```

Dòng này cho thấy câu trả lời có grounding, không phải LLM tự bịa.

## 8. Cách nói khi demo output

Ở đây câu hỏi là về phases của AI training program. Hệ thống đầu tiên embed câu hỏi, sau đó tìm top-3 chunks gần nhất. Ta thấy cả 3 chunks đều đến từ file AI Handbook, và chunk đầu tiên có score 0.742, khá cao. Sau đó agent đưa các chunks này vào prompt cho Gemini 2.5 Flash. Câu trả lời cuối cùng có 3 giai đoạn và có citation `[Source 1]`, `[Source 2]`, nên mình có thể trace lại nguồn.

## 9. Điểm mạnh của dự án

- Có pipeline RAG hoàn chỉnh.
- Có chunking strategy phù hợp với tài liệu policy.
- Có embedding thật bằng Gemini.
- Có source citation.
- Có benchmark queries và gold answers.
- Có metadata để filter/search tốt hơn.
- Test suite pass `42/42`.

## 10. Hạn chế có thể nói nếu bị hỏi

- Vector store hiện là in-memory, chưa persist database.
- Chưa có UI, demo bằng command line.
- Metadata filter chưa tự động suy luận từ câu hỏi trong `main.py`, chủ yếu dùng trong benchmark/code.
- Chất lượng phụ thuộc vào embedding model và cách chunk tài liệu.

## 11. Câu kết demo

Điểm em học được là RAG không chỉ là gọi LLM, mà quan trọng nhất là data pipeline phía trước: chọn tài liệu, chunking, embedding, metadata và đánh giá retrieval. Khi retrieval lấy đúng context thì LLM trả lời chính xác và có thể kiểm chứng bằng source.
