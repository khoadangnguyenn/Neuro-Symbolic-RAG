# 🎯 Neuro-Symbolic RAG EXACT Pipeline

This project implements a highly optimized, state-of-the-art **Neuro-Symbolic RAG** system designed to solve Physics and Logic reasoning tasks deterministically. By mapping Logic and Physics domain knowledge as a topological network and reasoning over it using logic/symbolic solvers (Z3 & SymPy), the pipeline completely eliminates typical LLM stochastic hallucinations. 

Equipped with Offline Pre-computation, Graph Traversal, Adaptive Routing, and an AST-guarded Secure Sandbox, the pipeline ensures safety, speed, and deterministic correctness while running strictly within the competition's **8B parameter limit**.

---

## 🧠 System Architecture



### 1. Hardware-Aware Adaptive Intent Router
A zero-LLM overhead router that classifies incoming queries instantly:
- **Complexity Assessment ($\kappa$)**: Uses `spaCy` (`en_core_web_sm`) and syntactic analysis (counting logical connectors and structure tokens) to evaluate query complexity.
- **Resource Pressure Monitoring ($R_p$)**: Evaluates CPU, MEM, and GPU utility levels.
- **Utility Optimization**: Determines whether to execute via the **Fast Path** (direct database/cache execution) or the **Hybrid Path** (with retrieval and generation) to optimize response speed and avoid system degradation.

### 2. HybridDB (Shared Knowledge Base)
Concept rules and formulas are stored across dual synchronized structures:
- **VectorDB (`ChromaDB`)**: Uses `BAAI/bge-small-en-v1.5` embeddings for semantic context search (with a robust lexical TF-IDF fallback if offline).
- **GraphDB (`NetworkX`)**: Captures structural causal relationships, using PageRank centrality to extract neighboring logic premises.
- **RAG-Based FOL Cache**: Automatically matches and retrieves previously translated FOL axioms from the VectorDB to bypass LLM translation steps.

### 3. Domain Reasoning Engines
- ⚡ **Type 1 (Logic) Pipeline**:
  - **Deterministic Horn Solver**: Evaluates Horn-compatible queries via forward chaining without LLM dependencies.
  - **Full Symbolic AST Path**: Translates complex logic questions into a compositional First-Order Logic (FOL) AST, repairs common translation errors using a post-processing sanitizer, and solves assertions using the **Z3 SMT Solver**.
- ⚡ **Type 2 (Physics) Pipeline**:
  - **Template-Based Fast Cache**: Replaces numbers with templates (e.g., `<NUM>`). Swaps new variables into cached execution scripts for instant answers.
  - **Jinja2 Code Generation**: Prompts the Main LLM to write SymPy scripts, applying an automated self-repair loop to fix syntax or execution trace errors on the fly.

### 4. Python Sandbox Executor
A secure runtime environment implementing:
- **AST Inspection**: Parses scripts to block hazardous modules (`os`, `sys`, `exec`, `eval`).
- **Resource Limits**: Enforces a strict `4.0s` timeout and restricts execution imports to trusted libraries (`math`, `sympy`, `z3`).

---

## 🚀 Quick Start

### Step 0: Download Model Weights
Create a `model/` directory and download the required GGUF weights:
```bash
mkdir -p model
# Download Main LLM (Qwen3-8B-GGUF)
wget https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf -O model/Qwen3-8B-Q8_0.gguf
```

### Step 1: Start the Single LLM Server
Run the local inference engine. Note that all Qwen3 thinking/reasoning modes are explicitly deactivated using the `--reasoning off` and `/no_think` switches to comply with latency constraints.
```bash
llama-server -m model/Qwen3-8B-Q8_0_gguf \
  --host 0.0.0.0 --port 8001 -c 16384 --alias exact-model \
  -ngl 99 --parallel 1 --flash-attn on
```

### Step 2: Initialize Database (Seeding)
Seed the HybridDB with formulas and logic rules. You only need to run this once. If it already exists, there’s no need to run it again.
```bash
EXACT_LLM_BASE_URL=http://localhost:8001 EXACT_LLM_MODEL=exact-model python3 scripts/auto_seeder.py
```

### Step 3: Start the API Gateway
Launch the Flask/stdlib endpoint container:
```bash
docker-compose up --build exact-api -d
```
The central API runs on port `8000`. The `/v1/models` endpoint is fully configured to route requests and verify model parameter constraints.

### Step 4: Tunnel Localhost to the Internet
Expose the endpoint port `8000` using ngrok or Cloudflare tunnels:
```bash
ngrok http 8000
```
This generates a public forwarding URL (e.g., `https://<random-id>.ngrok-free.app`).

### Step 5: Test the API Endpoint
Send a sample logic payload to test predictions:
```bash
curl -X POST <YOUR_NGROK_URL>/predict \
  -H "Content-Type: application/json" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{
    "query_id": "T1_0001",
    "type": "type1",
    "query": "Is Student A eligible for graduation?",
    "premises": [
      "A student who has completed at least 120 credits is eligible for graduation.",
      "Student A has completed 118 credits."
    ],
    "options": ["Yes", "No", "Uncertain"]
  }'
```
Or view the visual dashboard using the streamlit interface (default port `8501`):
```bash
python3 -m streamlit run exact_pipeline/app.py
```

### Step 6: Prepare `urls.txt` for Submission
Save your tunneling URLs in a `urls.txt` file inside your submission ZIP file:
```text
<YOUR_NGROK_URL>/predict
<YOUR_NGROK_URL>/v1/models
```

---

## 📂 Core Folder Structure

```text
exact_pipeline/
├── EXACT_Pipeline_Report.md     # Solution & Model Size report
├── README.md                    # Project walkthrough
├── Dockerfile                   # Python environment setup
├── docker-compose.yml           # Multi-container orchestration
├── requirements.txt             # Python libraries
├── app.py                       # Streamlit UI Dashboard
│
├── api/
│   └── server.py                # HTTP API router (/predict, /v1/models)
│
├── core/
│   ├── config.py                # System settings and environment loader
│   ├── data.py                  # Dataset loaders and normalizers
│   └── models.py                # Shared Pydantic data schemas
│
├── engines/
│   ├── executors.py             # Secure Z3/Python Sandbox Executors
│   ├── fol_ast.py               # First-Order Logic AST JSON schemas & builders
│   ├── fol_parser.py            # String-to-AST parser for prefix token streams
│   ├── horn_reasoner.py         # Forward-chaining deterministic solver
│   ├── logic.py                 # Logic reasoning orchestrator & sanitizers
│   ├── physics.py               # Physics formula parser & self-repair loop
│   ├── schema_learner.py        # Predicate schema mapping & dynamic aliases
│   └── symbolic_solver.py       # SMT constraint compiler for Z3
│
├── knowledge/
│   ├── graph_db.py              # Subgraph extractors, Steiner Tree, PageRank
│   ├── knowledge.py             # Shared HybridDB controller & index
│   └── retrieval.py             # VectorDB search (ChromaDB + lexical fallback)
│
├── llm/
│   ├── llm.py                   # OpenAI-compatible client (with Qwen3 thinking bypass)
│   └── templates.py             # System & Jinja2 prompt templates
│
├── orchestration/
│   ├── feedback.py              # Rule feedback extractors & GraphDB updates
│   ├── pipeline.py              # Unified entry point for solver routing
│   └── router.py                # Hardware-aware EMA utility routing
│
├── dataset/                     # Storage for VectorDB, GraphML files, and raw data
├── scripts/                     # Seeders and evaluation tools
├── tests/                       # Unit tests, smoke tests, and diverse testcases
└── utils/                       # Common string normalization utilities
```
