# Enhancing Reasoning and Search Capabilities of Language Agents Through Self-Evolving Search Agents

**Khóa luận tốt nghiệp Đại học - Trí tuệ Nhân tạo**  
**Sinh viên:** Vũ Minh Đức  
**Giảng viên hướng dẫn:** TS. Trần Hồng Việt  
**Năm:** 2026

---

## Mục tiêu

Xây dựng hệ thống thực nghiệm để minh họa và đánh giá ý tưởng **Self-Evolving Search Agents** (lấy cảm hứng từ paper *Dr. Zero*). Hệ thống so sánh 3 cấu hình:

1. **No Search Agent** (baseline)
2. **Search Agent** (TF-IDF retrieval + solver)
3. **Proposed Self-Evolving Search Agent** (synthetic data + curriculum filtering + iterative improvement simulation)

Kết quả thực nghiệm cho thấy retrieval cải thiện rõ rệt chất lượng, và cơ chế self-evolving (thông qua synthetic data + curriculum) tiếp tục mang lại cải thiện thêm.

> **Lưu ý**: Phiên bản hiện tại sử dụng **TF-IDF + rule-based solver** nhằm đảm bảo tính reproducible và dễ chạy demo khi bảo vệ (không phụ thuộc API key). Phiên bản đầy đủ sử dụng LLM (Groq) đang được phát triển song song.

---

## Kiến trúc tổng thể
src/
├── retriever.py          # TF-IDF Retriever
├── solver.py             # NoSearchSolver + SearchAugmentedSolver
├── proposer.py           # Sinh câu hỏi synthetic
├── evaluator.py          # Tính EM, Retrieval Hit, v.v.
├── curriculum.py         # Difficulty-based filtering
├── visualization.py
└── utils.py
experiments/
├── run_baseline.py
├── run_search.py
├── run_proposed.py
└── analyze_results.py
