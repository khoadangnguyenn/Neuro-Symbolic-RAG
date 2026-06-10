# EXACT 2026 Full Pipeline

![EXACT Pipeline Architecture](Full-Pipeline-Exact-2026.png)

Một giải pháp tiên tiến **Neuro-Symbolic AI** cho bài toán **The 2nd International XAI Challenge for Transparent Educational Question-Answering**. 

Pipeline này không sử dụng RAG văn bản (BM25) truyền thống dễ gây "ảo giác" (Hallucination). Thay vào đó, nó xây dựng một **HybridDB** (kết hợp VectorDB và GraphDB) để lập bản đồ tri thức Toán/Lý/Logic dưới dạng Mạng lưới liên kết (Topology). Phiên bản mới nhất đã được nâng cấp toàn diện với **Adaptive Routing** và **Local Query Expansion**.

---

## Kiến Trúc Neuro-Symbolic (Đã Nâng Cấp)

Luồng xử lý chính (Workflow):

1. **Adaptive Intent Router (Zero-LLM Overhead):** 
   Bộ định tuyến phần cứng nhận thức (Hardware-Aware). Sử dụng NLP tĩnh (`spaCy` mô hình `en_core_web_sm` chỉ 15MB) để chấm điểm độ phức tạp của câu hỏi ($\kappa$) và đo lường áp lực tài nguyên máy tính ($R_p$). Quyết định đi đường `Fast Path` hay `Hybrid Path` trong tíc tắc mà không cần gọi LLM, bảo vệ máy tính khỏi OOM (Sập RAM hoặc vRAM).
   
2. **HybridDB (Shared Knowledge Base):**
   Mỗi công thức hoặc quy luật được lưu trữ ở 2 dạng:
   - **VectorDB (ChromaDB):** Đã nâng cấp lên mô hình nhúng **`BAAI/bge-small-en-v1.5`** siêu nhẹ (~130MB RAM) nhưng có khả năng bắt ngữ nghĩa xuất sắc.
   - **GraphDB (NetworkX):** Xử lý cấu trúc nhân quả và topology (PageRank).

3. **Luồng Xử Lý Logic (Type 1) & Vật Lý (Type 2):**
   - **Fast Path:** Tra cứu trực tiếp quy luật bằng VectorDB. Nếu độ tự tin siêu cao, Bypass LLM và chạy thẳng công thức tĩnh.
   - **Hybrid RAG Path (Có Query Expansion):** Nếu Router quyết định dùng RAG, hệ thống sẽ kích hoạt **Query Expansion** bằng cách gọi một model siêu nhẹ (như **Gemma 1B**) để trích xuất các Keyword ẩn trước khi vào VectorDB. Việc gọi Query Expansion CHỈ xảy ra ở nhánh Hybrid, giúp bảo vệ hoàn toàn Fast Path khỏi sự lạm phát điểm độ phức tạp. Sau đó, hệ thống duyệt đồ thị, gom công thức và tiêm **Hàng rào Toạ độ (Coordinate Guardrail)** để cấm LLM đoán bừa khoảng cách hình học, ép giải bằng hệ toạ độ Oxy.
   - Cuối cùng, giao việc cho LLM chính (hoặc Gemma 1B) sinh mã Python lập trình.

4. **Python Sandbox Executor:**
   Mã do LLM sinh ra bị cô lập hoàn toàn (cấm `os`, `sys`, `exec()`, `eval()`), ép thời gian chạy (4.0s) và giới hạn chỉ dùng thư viện toán học an toàn (`math`, `sympy`, `z3`). Đảm bảo kết quả chính xác tuyệt đối.

---

## Chạy Nhanh Toàn Bộ Hệ Thống

### 1. Khởi tạo Cơ Sở Dữ Liệu (Seeding)
Hệ thống cần đọc file gốc (`Logic_Based_Educational_Queries.json` và `Physics_Problems_Text_Only.csv`) để tự động bóc tách và đúc thành các Node trong GraphDB.
*Lưu ý: Bạn chỉ cần chạy lệnh này 1 lần duy nhất.*
```bash
EXACT_LLM_BASE_URL=http://localhost:8001 EXACT_LLM_MODEL=exact-model python3 scripts/auto_seeder.py
```

### 2. Khởi động Máy chủ API & LLM
Khởi động API Server (chạy ở cổng 8000).
```bash
docker-compose up --build exact-api -d
```

**Tự Host Đa Mô Hình (Dual-LLM Architecture):**
Hệ thống hiện tại được thiết kế chạy **2 mô hình LLM chuyên biệt** song song ở ngoài Docker (Local Host):
- **LLM Chính (Giải toán/Code Z3, Sympy):** Chạy Qwen 2.5 7B.
- **LLM Trợ lý (Query Expansion):** Chạy Gemma 1B siêu nhẹ để bóc tách từ khóa.

Mở 2 terminal và chạy 2 `llama-server` ở 2 port khác nhau (ví dụ 8001 và 8002):

*Terminal 1 (Main LLM - Port 8001):*
```bash
llama-server -m model/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8001 -c 8192 --alias exact-model \
  -ngl 99 --parallel 1 --flash-attn on
```

*Terminal 2 (Expansion LLM - Port 8002):*
```bash
llama-server -m model/gemma-3-1b-it-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8002 -c 8192 --alias exact-model \
  -ngl 99 --parallel 1 --flash-attn on
```

*(Lưu ý: Nếu bạn chạy Python API trực tiếp, hãy gán biến `EXACT_LLM_BASE_URL=http://localhost:8001` và `EXACT_EXPANSION_LLM_BASE_URL=http://localhost:8002`)*

---

## Gọi API (Testing)

Endpoint chính để giải toán: `POST http://localhost:8000/answer`

**Ví dụ Gửi Câu Hỏi Vật Lý:**
```bash
curl -s http://localhost:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{
    "query_type": "type2",
    "question": "Two point charges q1 = 10^-8 C and q2 = -2×10^-8 C are placed in air at two points A and B, 8 cm apart. Calculate the net force."
  }'
```

**Hoặc chạy file test có sẵn:**
```bash
python3 test_custom.py
```

---

## Cấu Trúc File & Thư Mục Quan Trọng

Dưới đây là sơ đồ thư mục của dự án sau khi tái cấu trúc:

```text
EXACT-Full-Pipeline/
├── Diagram/                     # Chứa sơ đồ kiến trúc
├── docs/                        # Chứa các tài liệu giải pháp (Solution) và khai báo dữ liệu
├── test_client.py               # Công cụ kiểm thử tương tác tự động
├── test_debug.py                # Script dùng để gọi API kiểm thử và debug
├── test_llm_direct.py           # Gọi thử nghiệm LLM trực tiếp
│
└── exact_pipeline/              # Mã nguồn cốt lõi của hệ thống Neuro-Symbolic
    ├── Full-Pipeline-Exact-2026.png # Hình ảnh sơ đồ pipeline
    ├── docker-compose.yml       # Cấu hình khởi chạy nhanh API
    ├── Dockerfile               # Tệp tin cấu hình đóng gói Docker
    ├── dataset/                 # Chứa dữ liệu gốc (JSON, CSV) và VectorDB/GraphDB
    ├── model/                   # Nơi lưu trữ trọng số mô hình LLM (.gguf)
    ├── scripts/                 # Các công cụ hỗ trợ
    │   ├── auto_seeder.py       # Kịch bản tự động nạp và cấy dữ liệu vào HybridDB
    │   └── evaluate_local.py    # Kịch bản đánh giá độ chính xác (Accuracy)
    ├── tests/                   # Kịch bản kiểm thử (Smoke test)
    │   ├── smoke_test.py        # Chạy kiểm tra nhanh hệ thống
    │   └── test_custom.py       # Gọi API cho câu hỏi tùy biến
    ├── core/                    # Cấu hình (config.py) và định dạng dữ liệu (models.py)
    ├── engines/                 # Chứa Logic (Z3) / Physics (SymPy) pipelines 
    │   └── executors.py         # Môi trường Python Sandbox cách ly bảo mật
    ├── knowledge/               # Thư mục phụ trách HybridDB
    │   ├── graph_db.py          # NetworkX GraphDB và hàm tính điểm PageRank + Vector
    │   └── retrieval.py         # VectorDB (ChromaDB)
    ├── llm/                     # Xử lý giao tiếp với LLM
    │   ├── llm.py               # HTTP Client gọi tới LLM (vLLM / llama.cpp / Ollama)
    │   └── templates.py         # Nơi chứa các System Prompt (Jinja2)
    └── orchestration/           
        ├── router.py            # Bộ định tuyến (Intent Router) chia luồng Logic/Physics
        └── pipeline.py          # API Server chính (FastAPI/Flask)
```
