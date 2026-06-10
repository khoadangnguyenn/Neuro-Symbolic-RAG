import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))
os.environ["EXACT_PROJECT_ROOT"] = os.path.abspath("exact_pipeline")
os.environ["EXACT_LLM_BASE_URL"] = "http://localhost:8001"
os.environ["EXACT_CODE_TIMEOUT"] = "10.0"

from exact_pipeline.orchestration.pipeline import ExactPipeline

def run_test():
    pipeline = ExactPipeline()
    payload = {
        "query_type": "type1",
        "premises-NL": [
            "If a drone lacks image stabilization, then it does not have a high-quality camera."
        ],
        "question": "Does a drone have a high-quality camera?"
    }

    start = time.time()
    try:
        # Patch the LLM chat method locally to see exactly what is returned
        original_chat_json = pipeline.logic.llm.chat_json
        original_chat = pipeline.logic.llm.chat
        
        def mock_chat_json(*args, **kwargs):
            print(f"[DEBUG] chat_json called with {kwargs.get('user_prompt')[:100]}...")
            res = original_chat_json(*args, **kwargs)
            print(f"[DEBUG] chat_json returned: {res}")
            return res
            
        def mock_chat(*args, **kwargs):
            print(f"[DEBUG] chat called with {kwargs.get('user_prompt')[:100]}...")
            res = original_chat(*args, **kwargs)
            print(f"[DEBUG] chat returned: {res}")
            return res
            
        pipeline.logic.llm.chat_json = mock_chat_json
        pipeline.logic.llm.chat = mock_chat

        result = pipeline.answer_result(payload)
        print(f"[TEST] Answer received in {time.time() - start:.2f} seconds.")
        print(f"Answer: {result.answer}")
        print(f"Source: {result.source}")
    except Exception as e:
        print(f"[TEST] Exception: {e}")

if __name__ == '__main__':
    run_test()
