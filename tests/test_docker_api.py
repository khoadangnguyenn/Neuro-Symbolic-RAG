import json
import time
import urllib.request
import urllib.error

def run_test():
    url = "http://localhost:8000/answer"
    payload = {
        "query_type": "Math & Logic",
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

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    print("[TEST] Sending request to exact-api at localhost:8000...")
    start_time = time.time()
    try:
        # Give it a 60 second timeout for testing, no need to wait 300s
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode())
            print(f"[TEST] Success! Time taken: {time.time() - start_time:.2f}s")
            print(f"Result: {json.dumps(result, indent=2)}")
    except urllib.error.URLError as e:
        print(f"[TEST] Error: {e}. Time taken: {time.time() - start_time:.2f}s")
        if isinstance(e.reason, TimeoutError) or 'timed out' in str(e.reason).lower():
            print("==> The server took longer than 60 seconds to respond.")
            print("==> This confirms the container is still running the old blocking code or the LLM is just too slow.")

if __name__ == '__main__':
    run_test()
