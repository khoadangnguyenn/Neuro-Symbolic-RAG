import json
import time
import urllib.request
import urllib.error

def run_test(num_premises):
    url = "http://localhost:8000/answer"
    
    # 36 premises
    all_premises = [
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
    ]
    
    payload = {
        "query_type": "Math & Logic",
        "premises-NL": all_premises[:num_premises],
        "question": "Based on the above premises, which is the fewest in the following premises?\nA. ∀x (G(x) → (S(x) ∧ C(x)))\nB. ∃x (¬O(x) ∧ R(x))\nC. ∀x (¬B(x) → C(x))\nD. ∀x (R(x) → ¬G(x))"
    }

    print(f"\n[TEST] Running EXACT Pipeline Test with {num_premises} premises...")
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    start_time = time.time()
    try:
        # Tăng timeout lên 600 giây (10 phút) để đảm bảo llama.cpp có đủ thời gian xử lý toàn bộ 36 câu lệnh Z3
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.loads(response.read().decode())
            print(f"[TEST] Success! Time taken: {time.time() - start_time:.2f}s")
            print(f"Result: {json.dumps(result, indent=2)}")
    except urllib.error.URLError as e:
        print(f"[TEST] Error: {e}. Time taken: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    # Test với số lượng câu hỏi nhỏ để xác nhận hệ thống chạy mượt (FOL dịch bình thường)
    run_test(5)
    
    # Test với số lượng 36 câu hỏi để kích hoạt cơ chế Skip FOL Translation
    run_test(36)
