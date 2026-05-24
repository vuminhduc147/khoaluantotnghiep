# Enhancing Reasoning and Search Capabilities of Language Agents Through Self-Evolving Search Agents

**Khóa luận tốt nghiệp Đại học - Trí tuệ Nhân tạo**  
**Sinh viên:** Vũ Minh Đức  
**Giảng viên hướng dẫn:** TS. Trần Hồng Việt  
**Trường:** Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội (VNU-UET)  
**Năm:** 2026

---

## 📌 Mục tiêu

Xây dựng hệ thống thực nghiệm để **minh họa và đánh giá** ý tưởng **Self-Evolving Search Agents** (lấy cảm hứng từ paper *Dr. Zero*).

Hệ thống so sánh các phương pháp sau:

| Phương pháp                    | Mô tả                                                                 | Mục đích đánh giá                  |
|--------------------------------|-----------------------------------------------------------------------|------------------------------------|
| **No Search Baseline**         | Trả lời trực tiếp mà không sử dụng retrieval                          | Baseline                           |
| **Search Agent (TF-IDF)**      | Truy xuất tài liệu bằng TF-IDF trước khi trả lời                      | Đánh giá tác động của retrieval    |
| **Self-Evolving Agent**        | Sử dụng synthetic data + curriculum filtering (3 rounds)              | Mô phỏng quá trình tự cải thiện    |
| **Proposed Self-Evolving**     | Sử dụng kiến thức từ synthetic data để mở rộng query trên tập test thật | Đánh giá hiệu quả trên dữ liệu thực |

---

## 🏗️ Cấu trúc dự án (thực tế)

```
khoaluantotnghiep/
├── data/
│   ├── corpus.json                 # Kho tri thức (đoạn văn Wikipedia/AI)
│   └── test_questions.json         # ~200 câu hỏi test (one-hop & multi-hop)
│
├── src/
│   ├── bm25_retriever.py           # BM25 Retriever (chưa được sử dụng chính)
│   ├── curriculum.py               # DifficultyBasedFilter
│   ├── evaluator.py                # SimpleEvaluator (có evaluate + evaluate_without_gold)
│   ├── main.py                     # Script chính chạy toàn bộ thực nghiệm
│   ├── proposer.py                 # QuestionProposer (sinh câu hỏi synthetic)
│   ├── retriever.py                # TfidfRetriever (đang được sử dụng)
│   └── solver.py                   # NoSearchSolver + SearchAugmentedSolver
│
├── scripts/
│   ├── build_real_dataset.py       # Script hỗ trợ xây dựng dữ liệu
│   └── plot_results.py             # Script vẽ biểu đồ
│
├── results/
│   ├── figures/                    # Thư mục chứa biểu đồ
│   ├── baseline_search_results.csv
│   ├── no_search_results.csv
│   ├── proposed_real_test_results.csv
│   ├── self_evolving_results.csv
│   ├── summary_results.csv
│   ├── synthetic_summary_results.csv
│   ├── search_by_question_type.csv
│   └── type_comparison_results.csv
│
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng dẫn chạy

### 1. Cài đặt

```bash
git clone https://github.com/vuminhduc147/khoaluantotnghiep.git
cd khoaluantotnghiep
pip install -r requirements.txt
```

### 2. Chạy toàn bộ thực nghiệm

Chỉ cần chạy file `main.py`:

```bash
python src/main.py
```

Script sẽ tự động thực hiện theo thứ tự:

1. `run_no_search_baseline()`
2. `run_baseline_search()`
3. `run_self_evolving_agent()` (3 rounds trên synthetic data)
4. `run_proposed_on_real_test()`
5. `summarize_results()`
6. `save_type_comparison()`

Tất cả kết quả sẽ được lưu vào thư mục `results/`.

---

## 🔬 Giải thích các Experiment

### 1. No Search Baseline
- Sử dụng `NoSearchSolver`
- Trả lời câu hỏi mà **không có** thông tin từ corpus
- Đo lường baseline khi không có retrieval

### 2. Search Agent (TF-IDF)
- Sử dụng `TfidfRetriever` + `SearchAugmentedSolver`
- Truy xuất Top-K đoạn văn liên quan trước khi trả lời
- Đánh giá tác động của retrieval augmentation

### 3. Self-Evolving Agent (Synthetic)
- Sử dụng `QuestionProposer` để sinh câu hỏi synthetic
- Áp dụng `DifficultyBasedFilter` để lọc câu hỏi mức độ **medium + hard**
- Chạy **3 rounds**, mỗi round sinh 30 câu hỏi
- Đánh giá bằng `evaluate_without_gold` (auto-scoring)

### 4. Proposed Self-Evolving trên Real Test
- Sinh 100 câu hỏi synthetic → lọc theo độ khó
- Trích xuất các từ quan trọng (từ dài > 5 ký tự) từ synthetic questions
- **Mở rộng query** của câu hỏi thật bằng các từ này trước khi retrieval
- Đây là cách mô phỏng "self-evolving" trên dữ liệu thực

---

## 📊 Kết quả

Sau khi chạy `main.py`, các file kết quả chính:

| File                              | Nội dung                                      |
|-----------------------------------|-----------------------------------------------|
| `summary_results.csv`             | Bảng tổng hợp metrics các phương pháp         |
| `type_comparison_results.csv`     | So sánh theo loại câu hỏi (one-hop vs multi-hop) |
| `synthetic_summary_results.csv`   | Kết quả self-evolving trên dữ liệu tổng hợp   |
| `search_by_question_type.csv`     | Hiệu suất Search Agent theo loại câu hỏi      |
| `figures/`                        | Biểu đồ so sánh (nếu có script vẽ)            |

---

## 🛠️ Công nghệ sử dụng

- **Python**
- **scikit-learn** (TF-IDF Vectorizer + cosine similarity)
- **pandas** + **tqdm**
- **rank-bm25** (đã cài nhưng chưa dùng chính trong `main.py`)

---

## 📈 Điểm mạnh hiện tại

- Code **reproducible** (không cần API key LLM)
- Dễ chạy demo khi bảo vệ khóa luận
- Đã có đầy đủ pipeline: Synthetic data → Curriculum filtering → Evaluation
- Có so sánh chi tiết theo loại câu hỏi

---

## 🔮 Future Work (Định hướng phát triển)

- [ ] Tích hợp **LLM** (Groq / Gemini) vào `solver.py` và `proposer.py`
- [ ] Sử dụng **BM25** thay vì TF-IDF (đã có file `bm25_retriever.py`)
- [ ] Thêm **Iterative Query Refinement** (viết lại query nếu retrieval kém)
- [ ] Mở rộng đánh giá trên **HotpotQA** và **2WikiMultihopQA**
- [ ] Thêm cơ chế **few-shot update prompt** giữa các round self-evolving
- [ ] Tách `main.py` thành các file riêng trong folder `experiments/`

---

## 📝 Trích dẫn

Nếu sử dụng code cho nghiên cứu hoặc tham khảo, vui lòng trích dẫn:

> Vũ Minh Đức. *Enhancing Reasoning and Search Capabilities of Language Agents Through Self-Evolving Search Agents*. Khóa luận tốt nghiệp, Trí tuệ Nhân tạo, VNU-UET, 2026.

---

**Sinh viên thực hiện:** Vũ Minh Đức  
**Giảng viên hướng dẫn:** TS. Trần Hồng Việt

---

*Cập nhật lần cuối: Tháng 5/2026*
