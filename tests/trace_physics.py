import sys
import time
import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("EXACT_LLM_BASE_URL", "http://localhost:8001")
os.environ.setdefault("EXACT_LLM_MODEL", "exact-model")
os.environ.setdefault("EXACT_EXPANSION_LLM_BASE_URL", "http://localhost:8002")
os.environ.setdefault("EXACT_EXPANSION_LLM_MODEL", "exact-model")

sys.path.append("/Users/nguyendangkhoa/Documents/EXACT-Full-Pipeline")

from exact_pipeline.orchestration.pipeline import ExactPipeline
from exact_pipeline.engines.physics import PhysicsPipeline
from exact_pipeline.knowledge.knowledge import get_reasoning_subgraph_context
import exact_pipeline.engines.physics as phys_mod

def main():
    print("\n" + "="*80)
    print("🔍 EXACT PIPELINE — NEURO-SYMBOLIC DEEP TRACER")
    print("="*80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    file_path = "physics.json"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_query_id = sys.argv[1] if len(sys.argv) > 1 else None

    testcases = [item for item in data if item.get("type") == "type2"]
    
    if target_query_id:
        filtered = [item for item in testcases if item.get("query_id") == target_query_id]
        if not filtered:
            print(f"❌ Could not find query_id '{target_query_id}' in {file_path}")
            return
        payload = filtered[0]
    else:
        if not testcases:
             print("❌ No type2 testcases found.")
             return
        payload = testcases[-1]

    print(f"\n[1] 📥 INITIAL INPUT")
    print(f"    Query ID: {payload.get('query_id')}")
    print(f"    Query: {payload.get('query')}")

    print("\n[INIT] Initializing Pipeline...")
    t0 = time.time()
    pipeline = ExactPipeline()
    physics_engine = pipeline.physics
    print(f"[INIT] Done in {time.time()-t0:.2f}s")

    # --- MONKEY PATCHING FOR TRACING ---
    
    original_orchestrate = physics_engine._orchestrate_query
    def trace_orchestrate(question):
        print(f"\n[2] 🧠 ORCHESTRATION STAGE (Gemma 1B)")
        res = original_orchestrate(question)
        print(json.dumps(res, indent=4))
        return res
    physics_engine._orchestrate_query = trace_orchestrate
    
    original_search = physics_engine.index.search
    def trace_search(*args, **kwargs):
        print(f"\n[3] 📚 RAG RETRIEVAL (Vector DB)")
        print(f"    Searching for semantic matches...")
        hits = original_search(*args, **kwargs)
        for i, hit in enumerate(hits):
            q_text = hit.item.question[:80].replace("\n", " ") if hasattr(hit.item, "question") else str(hit.item)[:80]
            print(f"    - Hit {i+1} (Score: {hit.score:.4f}): {q_text}...")
        return hits
    physics_engine.index.search = trace_search
    
    original_get_subgraph = phys_mod.get_reasoning_subgraph_context
    def trace_get_subgraph(query, index, max_cards=4, max_depth=2):
        print(f"\n[4] 🕸️  GRAPH-DB SYMBOLIC RETRIEVAL (HybridDB)")
        print(f"    Extracting reasoning subgraph for constraints & laws...")
        context = original_get_subgraph(query, index, max_cards, max_depth)
        for line in context:
            print(f"    | {line}")
        return context
    phys_mod.get_reasoning_subgraph_context = trace_get_subgraph
    
    original_chat = physics_engine.llm.chat
    def trace_chat(*args, **kwargs):
        print(f"\n[5] 🤖 NEURO-SYMBOLIC SYNTHESIS (LLM to SymPy)")
        print(f"\n    --- System Prompt (Showing extracted laws) ---")
        system_prompt = kwargs.get('system_prompt', '')
        lines = system_prompt.split('\n')
        # Print the last 25 lines which contain the retrieved laws
        for line in lines[-25:]:
            print(f"    {line}")
            
        print(f"\n    --- Awaiting LLM Code Generation ---")
        res = original_chat(*args, **kwargs)
        print(f"\n    --- LLM JSON Output ---")
        print("    " + "\n    ".join(json.dumps(res, indent=2).split('\n')))
        return res
    physics_engine.llm.chat = trace_chat

    original_executor_run = physics_engine.executor.run
    def trace_executor(code):
        print(f"\n[6] ⚡ SYMBOLIC SOLVER EXECUTION (Python Sandbox / SymPy)")
        print(f"    Executing generated code:")
        print("    ----------------------------------------")
        print("    " + "\n    ".join(code.split('\n')))
        print("    ----------------------------------------")
        res = original_executor_run(code)
        print(f"\n    --- Execution Output ---")
        print(f"    Success: {res.ok}")
        print(f"    Stdout: {res.stdout.strip()}")
        if res.error:
            print(f"    Error: {res.error.strip()}")
        print(f"    Parsed Answer: {res.answer}")
        return res
    physics_engine.executor.run = trace_executor

    # --- END MONKEY PATCHING ---
    
    print("\n" + "-"*80)
    print(f"🏃 RUNNING TRACE...")
    print("-" * 80)
    
    try:
        # Override cache to ensure execution runs
        physics_engine.code_cache = {}
        
        results = pipeline.answer(payload)
        for idx, result in enumerate(results):
            print(f"\n[7] 🎯 FINAL RESULT")
            print(f"    Answer: {result.get('answer')}")
            print(f"    Explanation: {result.get('explanation')}")
            print(f"    Confidence: {result.get('confidence')}")
            print(f"    Source: {result.get('source')}")
    except Exception as e:
        print(f"❌ Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
