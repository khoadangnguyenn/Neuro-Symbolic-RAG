import urllib.request
import json
import time
import sys

# Define diverse testcases
# Type 1 = Logic (Math & Logic)
# Type 2 = Physics (Physics & Scholarship)
TESTCASES = [
    # ==========================
    # TYPE 1: LOGIC (Math & Logic)
    # ==========================
    {
        "name": "Type 1 - Logic (Yes/No/Uncertain) - Simple Entailment",
        "payload": {
            "query_type": "type1",
            "premises-NL": [
                "All cats are mammals.",
                "Fluffy is a cat."
            ],
            "question": "Is Fluffy a mammal?",
            "options": ["Yes", "No", "Uncertain"]
        },
        "expected_answer": "Yes"
    },
    {
        "name": "Type 1 - Logic (Choose True/Strongest)",
        "payload": {
            "query_type": "type1",
            "premises-NL": [
                "If it rains, the ground gets wet.",
                "It is raining today."
            ],
            "question": "Based on the premises, which statement is true?\nA. The ground is dry\nB. The ground is wet\nC. It is snowing"
        },
        "expected_answer": "B"
    },
    {
        "name": "Type 1 - Logic (Contradiction / Modus Tollens)",
        "payload": {
            "query_type": "type1",
            "premises-NL": [
                "If a drone has a high-quality camera, it has long battery life.",
                "Drone X does not have long battery life."
            ],
            "question": "Based on the above premises, which of the following is true?\nA. Drone X has a high-quality camera.\nB. Drone X does not have a high-quality camera."
        },
        "expected_answer": "B"
    },
    {
        "name": "Type 1 - Logic (Fewest Premises)",
        "payload": {
            "query_type": "type1",
            "premises-NL": [
                "If A then B.",
                "If B then C.",
                "If A then C.",
                "A is true."
            ],
            "question": "Based on the above premises, which is the fewest in the following premises?\nA. B is true\nB. C is true\nC. D is true"
        },
        "expected_answer": "A"  
    },
    {
        "name": "Type 1 - Logic (FOL Symbolic Logic)",
        "payload": {
            "query_type": "type1",
            "premises-FOL": [
                "∀x (C(x) → M(x))",
                "C(Fluffy)"
            ],
            "question": "Is Fluffy a mammal?\nA. Yes\nB. No\nC. Uncertain"
        },
        "expected_answer": "A"
    },

    # ==========================
    # TYPE 2: PHYSICS (Physics & Scholarship)
    # ==========================
    {
        "name": "Type 2 - Physics Calculation (Kinematics)",
        "payload": {
            "query_type": "type2",
            "question": "A car travels 100 km in 2 hours. What is its average speed in km/h?\nA. 30\nB. 40\nC. 50\nD. 60"
        },
        "expected_answer": "50"
    },
    {
        "name": "Type 2 - Physics Calculation (Electrostatics)",
        "payload": {
            "query_type": "type2",
            "question": "Two point charges q1 = 10^-8 C and q2 = -2×10^-8 C are placed in air at two points A and B, 8 cm apart. Calculate the net force."
        },
        "expected_answer": "" # Open ended physics calculation
    },
    {
        "name": "Type 2 - Physics Formula Extraction",
        "payload": {
            "query_type": "type2",
            "question": "What is the formula for calculating the kinetic energy of an object given its mass and velocity?\nA. E = mc^2\nB. K = 1/2 m v^2\nC. F = ma\nD. P = mv"
        },
        "expected_answer": "B"
    }
]

def run_tests():
    print("==================================================")
    print("🚀 EXACT PIPELINE DIVERSE TEST SUITE")
    print("==================================================\n")
    
    passed = 0
    total = len(TESTCASES)
    results = []

    for idx, test in enumerate(TESTCASES, 1):
        name = test["name"]
        payload = test["payload"]
        expected = test.get("expected_answer", "N/A")
        
        print(f"[{idx}/{total}] Running: {name}")
        print(f"    📝 Question: {payload['question'].split(chr(10))[0][:80]}...")
        
        req = urllib.request.Request(
            "http://localhost:8000/answer",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                result_data = json.loads(response.read().decode('utf-8'))
                
                # Check for single dict or list
                if isinstance(result_data, list) and len(result_data) > 0:
                    result_data = result_data[0]
                    
                answer = str(result_data.get('answer', '')).strip()
                duration = time.time() - t0
                
                # Check if answer contains expected (simple substring match for robustness)
                # If expected is empty, any non-error response passes
                if not expected:
                    is_pass = True
                else:
                    is_pass = expected.lower() in answer.lower()
                    
                status = "✅ PASS" if is_pass else "❌ FAIL"
                
                if is_pass:
                    passed += 1
                    
                print(f"    {status} (⏱️ {duration:.2f}s) | Expected: {expected} | Got: {answer}")
                
                # If fail, print more details
                if not is_pass:
                    print(f"    🔍 Explanation: {result_data.get('explanation', '')}")
                    if "execution_errors" in result_data.get("metadata", {}):
                        print(f"    ⚠️ Errors: {result_data['metadata']['execution_errors']}")
                        
                results.append({
                    "name": name,
                    "status": status,
                    "duration": duration,
                    "expected": expected,
                    "got": answer
                })
                
        except Exception as e:
            duration = time.time() - t0
            print(f"    💥 ERROR (⏱️ {duration:.2f}s): {str(e)}")
            results.append({
                "name": name,
                "status": "💥 ERROR",
                "duration": duration,
                "expected": expected,
                "got": str(e)
            })
            
        print("-" * 50)

    # Print Summary
    print("\n==================================================")
    print(f"📊 TEST SUMMARY: {passed}/{total} PASSED")
    print("==================================================")
    for r in results:
        print(f"{r['status']} | {r['name']} ({r['duration']:.2f}s)")
        if r['status'] != "✅ PASS" and r['expected']:
            print(f"         Expected: {r['expected']} | Got: {r['got']}")

if __name__ == "__main__":
    run_tests()
