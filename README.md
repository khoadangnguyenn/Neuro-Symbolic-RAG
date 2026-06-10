# 🎯 Neuro-Symbolic RAG
This project completely replaces traditional text-based RAG (BM25) to eliminate LLM hallucinations. It leverages a Neuro-Symbolic HybridDB to map Logic and Physics knowledge as a topological network, heavily augmented with Offline Pre-computation, Graph Traversal (Steiner Tree), and Adaptive Routing.

## 🧠 System Architecture
![Pipeline Optimized](Pipeline-Optimized.png)

The pipeline consists of 4 core components:

### 1. Adaptive Intent Router (Zero-LLM Overhead)
A hardware-aware router that categorizes queries instantly without consuming LLM resources:
- **Static NLP**: Uses `spaCy` (`en_core_web_sm`) to calculate query complexity ($\kappa$) and measure computational pressure ($R_p$).
- **Dynamic Routing**: Automatically routes queries to the **Fast Path** or **Hybrid Path** to optimize speed and prevent OOM issues.

### 2. HybridDB (Shared Knowledge Base)
Formulas and rules are stored across two parallel formats:
- **VectorDB (ChromaDB)**: Uses `BAAI/bge-small-en-v1.5` for lightning-fast semantic retrieval.
- **GraphDB (NetworkX)**: Maps causal structures and topologies, utilizing PageRank for knowledge ranking.

### 3. Execution Paths
- ⚡ **Fast Path**: Direct VectorDB lookup. Bypasses the LLM entirely and executes static code if the confidence score is high.
- 🔍 **Hybrid RAG Path**: Uses a secondary model (Gemma 1B) for keyword extraction (Query Expansion) prior to retrieval. Includes Coordinate Guardrails to prevent spatial hallucinations.
- 💻 **Code Generation**: The Main LLM generates precise Python/Z3/SymPy code to solve the extracted problem.

### 4. Python Sandbox Executor
A completely isolated execution environment:
- **Security**: Blocks hazardous modules (`os`, `sys`, `exec`, `eval`).
- **Constraints**: Enforces a strict 4.0s timeout and only permits math libraries (`math`, `sympy`, `z3`).



---

## 🚀 Quick Start

The system relies on a **Dual-LLM** architecture. Please execute the following steps in order.

### Step 0: Download Models
Create a `model/` directory and download the required GGUF weights (e.g., from Hugging Face):

```bash
mkdir -p model

# Download Main LLM (Qwen2.5 7B)
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf -O model/Qwen2.5-7B-Instruct-Q4_K_M.gguf

# Download Expansion LLM (Gemma 3 1B)
wget https://huggingface.co/bartowski/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf -O model/gemma-3-1b-it-Q4_K_M.gguf
```

### Step 1: Start the Dual-LLM Servers
Run two LLM processes locally in separate terminals:

**Terminal 1 (Main LLM - Math & Z3/SymPy Code Generation):**
```bash
llama-server -m model/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8001 -c 8192 --alias exact-model \
  -ngl 99 --parallel 1 --flash-attn on
```

**Terminal 2 (Expansion LLM - High-speed Orchestration):**
```bash
llama-server -m model/gemma-3-1b-it-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8002 -c 8192 --alias exact-model \
  -ngl 99 --parallel 1 --flash-attn on
```

### Step 2: Initialize the Database (Seeding)
*(Note: Run this only once to build the GraphDB nodes, if you change the items in VectorDB, you need to run this again to update the GraphDB. Otherwise, you can skip this step)*
```bash
EXACT_LLM_BASE_URL=http://localhost:8001 EXACT_LLM_MODEL=exact-model python3 scripts/auto_seeder.py
```

### Step 3: Start the API Server
Deploy the main API Server (Port 8000) via Docker.
```bash
docker-compose up --build exact-api -d
```
*(For local Python execution without Docker, export: `EXACT_LLM_BASE_URL=http://localhost:8001` and `EXACT_EXPANSION_LLM_BASE_URL=http://localhost:8002`)*

---

## 🎯 API Testing

**Main Endpoint:** `POST http://localhost:8000/answer`

**Method 1: Direct cURL (Physics Query Example):**
```bash
curl -s http://localhost:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{
    "query_type": "type2",
    "question": "Two point charges q1 = 10^-8 C and q2 = -2×10^-8 C are placed in air at two points A and B, 8 cm apart. Calculate the net force."
  }'
```

**Method 2: Run the local test script:**
```bash
python3 test_custom.py
```
**Method 3: Open Web UI**
```bash
python3 -m streamlit run app.py
```

---

## 📂 Core Folder Structure

```text
EXACT-Full-Pipeline/
├── Diagram/                     # System architecture diagrams
├── docs/                        # Solution documentation
├── test_client.py               # Automated interaction client
├── test_debug.py                # Debug API calls
├── test_llm_direct.py           # Direct LLM connection tests
│
└── exact_pipeline/              # MAIN SOURCE CODE
    ├── Full-Pipeline-Exact-2026.png 
    ├── docker-compose.yml       # Docker deployment config
    ├── Dockerfile               # Environment package
    ├── dataset/                 # Raw Data & VectorDB/GraphDB storage
    ├── model/                   # LLM weights (.gguf)
    │
    ├── core/                    # Configs & Pydantic models
    ├── engines/                 
    │   └── executors.py         # Python Sandbox (Isolated execution)
    ├── knowledge/               # HybridDB processors
    │   ├── graph_db.py          # NetworkX: Topology, causality, PageRank
    │   └── retrieval.py         # ChromaDB: Vector Extraction
    ├── llm/                     
    │   ├── llm.py               # HTTP Client for vLLM/llama.cpp
    │   └── templates.py         # System Prompts (Jinja2)
    ├── orchestration/           
    │   ├── router.py            # Intent Router (Logic/Physics)
    │   └── pipeline.py          # FastAPI/Flask Server Init
    │
    ├── scripts/                 
    │   ├── auto_seeder.py       # HybridDB Auto-seeder
    │   └── evaluate_local.py    # Local Accuracy evaluation
    └── tests/                   
        ├── smoke_test.py        # Module quick tests
        └── test_custom.py       # Custom query API caller
```