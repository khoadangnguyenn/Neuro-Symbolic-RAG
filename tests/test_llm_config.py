import sys
import os

sys.path.insert(0, os.path.abspath("."))
os.environ["EXACT_PROJECT_ROOT"] = os.path.abspath("exact_pipeline")
os.environ["EXACT_LLM_BASE_URL"] = "http://localhost:8001"

from exact_pipeline.orchestration.pipeline import ExactPipeline

def run_test():
    pipeline = ExactPipeline()
    print("Logic pipeline LLM enabled:", pipeline.logic_pipeline.llm.enabled)
    print("Logic pipeline LLM URL:", pipeline.logic_pipeline.llm.base_url)

if __name__ == '__main__':
    run_test()
