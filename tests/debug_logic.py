import sys
import time
import json
import logging
import re
import os
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("EXACT_LLM_BASE_URL", "http://localhost:8001")
os.environ.setdefault("EXACT_LLM_MODEL", "exact-model")
os.environ.setdefault("EXACT_EXPANSION_LLM_BASE_URL", "http://localhost:8002")
os.environ.setdefault("EXACT_EXPANSION_LLM_MODEL", "exact-model")

sys.path.append("/Users/nguyendangkhoa/Documents/EXACT-Full-Pipeline")

from exact_pipeline.orchestration.pipeline import ExactPipeline

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "="*60)
    print("🚀 EXACT PIPELINE — CHUNKED TRANSLATION TRACER (v3)")
    print("="*60)
    
    with open("tests/test_single_drone.json", "r") as f:
        payload = json.load(f)
    
    question = payload["question"]
    premises_nl = payload.get("premises-NL", [])

    print(f"\n📥 Input: {len(premises_nl)} premises")
    print(f"📥 Question: {question[:100]}...")

    print("\n[INIT] Initializing Pipeline...")
    t0 = time.time()
    pipeline = ExactPipeline()
    logic_engine = pipeline.logic
    print(f"[INIT] Done in {time.time()-t0:.2f}s")

    # =========================================================================
    # STEP 1: ORCHESTRATION (Gemma 3)
    # =========================================================================
    matches = re.findall(r"(?:^|\n)([A-Z]\.)\s+", question)
    options = matches if len(matches) >= 2 else []
    
    from exact_pipeline.engines.logic import classify_logic_query_type
    query_type = classify_logic_query_type(options)
    
    print("\n" + "-"*60)
    print(f"🧠 STEP 1: ORCHESTRATION (Gemma 3) — query_type: {query_type}")
    print("-"*60)
    
    t0 = time.time()
    orchestration_data = logic_engine._orchestrate_query(question, query_type)
    print(f"[⏱️] Time: {time.time()-t0:.2f}s")
    print(f"Output:\n{json.dumps(orchestration_data, indent=2, ensure_ascii=False)}")

    # =========================================================================
    # STEP 2: VECTOR RETRIEVAL
    # =========================================================================
    print("\n" + "-"*60)
    print("🔍 STEP 2: VECTOR RETRIEVAL (ChromaDB)")
    print("-"*60)
    t0 = time.time()
    complexity = orchestration_data.get("complexity_score", 3)
    vector_k = max(5, int(complexity * 10))
    search_query = question + "\n" + "\n".join(orchestration_data.get("semantic_anchors", []))
    hits = logic_engine.index.search(search_query, k=vector_k, reranker=logic_engine.reranker, rerank_top_k=2)
    print(f"[⏱️] Time: {time.time()-t0:.2f}s — {len(hits)} hits")

    # =========================================================================
    # STEP 3 & 4: CHUNKED FOL TRANSLATION -> Z3 SOLVER
    # =========================================================================
    print("\n" + "-"*60)
    print(f"📝 STEP 3: CHUNKED FOL TRANSLATION & POST-PROCESSING")
    print("-"*60)
    
    fol_results = None
    
    t0 = time.time()
    try:
        result = logic_engine._answer_with_symbolic_logic(
            question=question,
            premises_nl=premises_nl,
            premises_fol=fol_results,
            hits=hits,
            orchestration_data=orchestration_data
        )
        print(f"\n[⏱️] Time: {time.time()-t0:.2f}s")
        if result:
            print(f"🎯 Verdict: {result.answer}")
            print(f"🎯 Premises Used: {result.premises_used}")
            print(f"🎯 Explanation: {result.explanation}")
        else:
            print("🎯 Result is None (Uncertain or Failed all retries)")
    except Exception as e:
        print(f"[⏱️] Time: {time.time()-t0:.2f}s")
        print(f"❌ _answer_with_symbolic_logic crashed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("📊 DONE")
    print("="*60)

if __name__ == '__main__':
    main()
