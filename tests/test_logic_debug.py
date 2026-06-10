import sys
import os
import time

# Ensure we can import exact_pipeline
sys.path.insert(0, os.path.abspath("."))
os.environ["EXACT_PROJECT_ROOT"] = os.path.abspath("exact_pipeline")
os.environ["EXACT_LLM_BASE_URL"] = "http://localhost:8001"

from exact_pipeline.orchestration.pipeline import ExactPipeline

def run_test():
    print("[TEST] Initializing Pipeline...")
    pipeline = ExactPipeline()
    
    payload = {
        "query_type": "type1",
        "premises-NL": [
            "If a drone lacks image stabilization, then it does not have a high-quality camera.",
            "Every drone has a long remote control range.",
            "There is at least one drone that has obstacle avoidance.",
            "If a drone does not have a long remote control range, then it does not have image stabilization.",
            "If a drone does not have a high-quality camera, then it does not have long battery life.",
            "If a drone has a high-quality camera, then it has a long remote control range.",
            "There is at least one drone that has a long remote control range.",
            "There is at least one drone that has GPS navigation.",
            "If a drone has GPS navigation, then it has obstacle avoidance.",
            "If a drone has image stabilization, then it has a long remote control range.",
            "Every drone has image stabilization.",
            "Every drone has a high-quality camera.",
            "If a drone has a long remote control range, then it has obstacle avoidance.",
            "There is at least one drone that has long battery life.",
            "If a drone has obstacle avoidance, then it has a long remote control range.",
            "Every drone has obstacle avoidance.",
            "There is at least one drone that has a high-quality camera.",
            "If a drone has long battery life, then it has a long remote control range.",
            "If a drone does not have a high-quality camera, then it does not have a long remote control range.",
            "If a drone does not have a high-quality camera, then it does not have GPS navigation.",
            "If a drone does not have GPS navigation, then it does not have a long remote control range.",
            "If having a long remote control range implies obstacle avoidance, then every drone has a long remote control range.",
            "If not having a high-quality camera implies not having a long remote control range, then every drone has obstacle avoidance.",
            "If having GPS navigation implies obstacle avoidance, then not having GPS navigation implies not having a long remote control range.",
            "If having long battery life implies a long remote control range, then there is at least one drone with long battery life.",
            "If not having a high-quality camera implies not having long battery life, then every drone has a long remote control range.",
            "If having a long remote control range implies obstacle avoidance, then if a drone has GPS navigation, then it has obstacle avoidance.",
            "If not having a long remote control range implies not having image stabilization, then if having a long remote control range implies obstacle avoidance, every drone has a long remote control range.",
            "If (if long battery life implies a long remote control range then there is at least one drone with long battery life) then if a long remote control range implies obstacle avoidance, every drone has a long remote control range.",
            "If there is at least one drone with a long remote control range, then there is at least one drone with GPS navigation.",
            "If having GPS navigation implies obstacle avoidance, then there is at least one drone with GPS navigation.",
            "If a drone does not have long battery life, then it does not have image stabilization.",
            "If a drone has image stabilization, then it has a high-quality camera.",
            "Every drone has GPS navigation.",
            "If a drone does not have a high-quality camera, then it does not have obstacle avoidance.",
            "Every drone has long battery life."
        ],
        "question": "Based on the above premises, which is the fewest in the following premises?\nA. ∀x (G(x) → (S(x) ∧ C(x)))\nB. ∃x (¬O(x) ∧ R(x))\nC. ∀x (¬B(x) → C(x))\nD. ∀x (R(x) → ¬G(x))"
    }

    print("[TEST] Calling pipeline.answer_result(payload)...")
    start = time.time()
    try:
        result = pipeline.answer_result(payload)
        print(f"[TEST] Answer received in {time.time() - start:.2f} seconds.")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence}")
        print(f"Source: {result.source}")
        if hasattr(result, 'metadata'):
            print(f"Metadata: {result.metadata}")
    except Exception as e:
        print(f"[TEST] Exception during pipeline.answer_result: {e}")

if __name__ == '__main__':
    run_test()
