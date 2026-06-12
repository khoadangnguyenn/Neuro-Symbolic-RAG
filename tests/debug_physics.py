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

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "="*60)
    print("🚀 EXACT PIPELINE — PHYSICS TRACER")
    print("="*60)
    
    # Change to exact_pipeline directory if not already there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    file_path = "physics.json"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_query_id = sys.argv[1] if len(sys.argv) > 1 else None

    # Filter to type2 and the specific query_id if provided
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

    print(f"\n📥 Query ID: {payload.get('query_id')}")
    print(f"📥 Type: {payload.get('type')}")
    print(f"📥 Query: {payload.get('query')}")

    print("\n[INIT] Initializing Pipeline...")
    t0 = time.time()
    pipeline = ExactPipeline()
    print(f"[INIT] Done in {time.time()-t0:.2f}s")

    print("\n" + "-"*60)
    print(f"🧠 RUNNING PIPELINE FOR PHYSICS")
    print("-"*60)
    
    t0 = time.time()
    try:
        results = pipeline.answer(payload)
        print(f"\n[⏱️] Total Execution Time: {time.time()-t0:.2f}s")
        for idx, result in enumerate(results):
            print(f"\n--- Result {idx+1} ---")
            print(f"🎯 Query ID: {result.get('query_id')}")
            print(f"🎯 Answer: {result.get('answer')}")
            print(f"🎯 Confidence: {result.get('confidence')}")
            print(f"🎯 Source: {result.get('source')}")
            print(f"🎯 Explanation:\n{result.get('explanation')}")
    except Exception as e:
        print(f"[⏱️] Time: {time.time()-t0:.2f}s")
        print(f"❌ Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
