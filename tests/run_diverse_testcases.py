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
            "question": "Based on the premises, which statement is true?\nA. The ground is dry\nB. The ground is wet\nC. It is snowing",
            "options": ["A", "B", "C"]
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
            "question": "Based on the above premises, which of the following is true?\nA. Drone X has a high-quality camera.\nB. Drone X does not have a high-quality camera.",
            "options": ["A", "B"]
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
            "question": "Based on the above premises, which is the fewest in the following premises?\nA. B is true\nB. C is true\nC. D is true",
            "options": ["A", "B", "C"]
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
            "question": "Is Fluffy a mammal?\nA. Yes\nB. No\nC. Uncertain",
            "options": ["A", "B", "C"]
        },
        "expected_answer": "A"
    },
    {
        "name": "Type 1 - Advanced Logic (Multi-step Modus Tollens with Quantifiers)",
        "payload": {
            "query_type": "yes_no_uncertain",
            "premises-NL": [
                "Every autonomous vehicle that passes the safety test is granted a deployment permit.",
                "No vehicle is granted a deployment permit unless it has certified LiDAR sensors.",
                "Vehicle Delta does not have certified LiDAR sensors.",
                "Vehicle Delta is currently equipped with standard radar."
            ],
            "question": "Does Vehicle Delta pass the safety test?",
            "options": ["Yes", "No", "Uncertain"]
        },
        "expected_answer": "No"
    },
    {
        "name": "Type 1 - Advanced Logic (Transitive Disjunction and Excluded Middle)",
        "payload": {
            "query_type": "multiple_choice",
            "premises-NL": [
                "A server hub is either energy-efficient or it requires continuous liquid cooling.",
                "If a server hub is energy-efficient, it utilizes next-gen ARM processors.",
                "If a server hub requires continuous liquid cooling, it utilizes high-voltage cooling pumps.",
                "Server Cluster Omega does not utilize next-gen ARM processors."
            ],
            "question": "Based on the premises, which statement is true?",
            "options": [
                "A. Server Cluster Omega does not require continuous liquid cooling.",
                "B. Server Cluster Omega utilizes high-voltage cooling pumps.",
                "C. Server Cluster Omega is energy-efficient.",
                "D. It is uncertain whether Server Cluster Omega utilizes high-voltage cooling pumps."
            ]
        },
        "expected_answer": "B"
    },
    {
        "name": "Type 1 - Advanced Logic (FOL with Complex Variable Binding & Multi-Argument Predicates)",
        "payload": {
            "query_type": "yes_no_uncertain",
            "premises-FOL": [
                "∀x (Researcher(x) → ∃y (LabAccess(x, y) ∧ SecureFacility(y)))",
                "∀x ∀y ((Researcher(x) ∧ LabAccess(x, y) ∧ SecureFacility(y)) → HasBiometricKey(x))",
                "Researcher(Asha)",
                "¬HasBiometricKey(Asha)"
            ],
            "question": "Do the premises prove that Asha has lab access to any secure facility?",
            "options": ["Yes", "No", "Uncertain"]
        },
        # The premises are classically inconsistent, so neither polarity is a
        # unique three-way answer.  The pipeline exposes the UNSAT status and
        # normalizes it to Uncertain when no contradiction option is offered.
        "expected_answer": "Uncertain"
    },
    {
        "name": "Type 1 - Advanced Logic (Invalid Inverse-Inference Trap)",
        "payload": {
            "query_type": "multiple_choice",
            "premises-NL": [
                "If an AI model is trained on medical data and passes the clinical trial, it is certified for hospital deployment.",
                "Any AI model that exhibits high variance in diagnostic accuracy cannot pass the clinical trial.",
                "Model Med-V4 is trained on medical data.",
                "Model Med-V4 exhibits high variance in diagnostic accuracy."
            ],
            "question": "Which conclusion is strictly supported by the premises?",
            "options": [
                "A. Model Med-V4 is certified for hospital deployment.",
                "B. Model Med-V4 passes the clinical trial but fails certification.",
                "C. Model Med-V4 cannot be certified for hospital deployment.",
                "D. Model Med-V4 does not exhibit high variance."
            ]
        },
        # C would deny the antecedent: (trained ∧ passes) -> certified and
        # ¬passes do not entail ¬certified.  No listed option is strictly
        # supported by the stated premises.
        "expected_answer": "Uncertain"
    },
    {
        "name": "Type 1 - Advanced Logic (Complex Chain with Biconditional Equivalence)",
        "payload": {
            "query_type": "yes_no_uncertain",
            "premises-NL": [
                "A smart greenhouse triggers autonomous watering if and only if the soil moisture drops below 30 percent.",
                "The soil moisture drops below 30 percent whenever the local heatwave sensor is active.",
                "The local heatwave sensor of Greenhouse Basil is currently active."
            ],
            "question": "Does Greenhouse Basil trigger autonomous watering according to the premises?",
            "options": ["Yes", "No", "Uncertain"]
        },
        "expected_answer": "Yes"
    },
    {
        "name": "Type 1 - Advanced Logic (Nested Quantifiers and Vacuous Truth Trap)",
        "payload": {
            "query_type": "multiple_choice",
            "premises-NL": [
                "All secure microgrids have at least one backup generator.",
                "Every backup generator in a secure microgrid must pass the weekly load test.",
                "The Harbor Microgrid has no backup generators.",
                "The Harbor Microgrid is a secure microgrid."
            ],
            "question": "What can be logically concluded about the Harbor Microgrid from the given premises?",
            "options": [
                "A. The Harbor Microgrid's backup generators passed the weekly load test.",
                "B. The premises contain a logical contradiction, making any statement provable (Unsat Core / Inconsistency).",
                "C. The Harbor Microgrid is secure but needs a generator.",
                "D. The Harbor Microgrid is not a microgrid."
            ]
        },
        "expected_answer": "B"
    },
  {
    "request_payload": {
      "options": [
        "Yes",
        "No",
        "Uncertain"
      ],
      "premises": [
        "Every autonomous drone that passes the safety navigation test is granted an operational waiver.",
        "No drone is granted an operational waiver unless it is equipped with a certified backup transceiver.",
        "Drone Kestrel-X does not have a certified backup transceiver.",
        "Drone Kestrel-X is currently painted high-visibility orange.",
        "The testing facility has 4 active runways."
      ],
      "query": "Does Drone Kestrel-X pass the safety navigation test?",
      "query_id": "hard_type1_modus_tollens",
      "type": "type1"
    },
    "expected": {
      "answer": "No",
      "premises_used": [
        0,
        1,
        2
      ]
    }
  },
  {
    "request_payload": {
      "options": [
        "Yes",
        "No",
        "Uncertain"
      ],
      "premises": [
        "An automated manufacturing node is either energy-efficient or it requires continuous refrigerant cooling.",
        "If a manufacturing node is energy-efficient, it utilizes RISC-V processing cores.",
        "If a manufacturing node requires continuous refrigerant cooling, it utilizes high-pressure hydraulic pumps.",
        "Node Delta-6 does not utilize RISC-V processing cores.",
        "Node Delta-6 was manufactured in 2025."
      ],
      "query": "Does Node Delta-6 utilize high-pressure hydraulic pumps?",
      "query_id": "hard_type1_disjunctive_syllogism",
      "type": "type1"
    },
    "expected": {
      "answer": "Yes",
      "premises_used": [
        0,
        1,
        2,
        3
      ]
    }
  },
  {
    "request_payload": {
      "options": [
        "Yes",
        "No",
        "Uncertain"
      ],
      "premises": [
        "All authenticated Edge nodes can broadcast telemetry data if they are within the gateway's coverage range.",
        "Node Echo is an authenticated Edge node.",
        "Node Echo is currently broadcasting telemetry data.",
        "Node Echo has a remaining battery capacity of 85 percent.",
        "No premise states whether Node Echo uses a directional antenna."
      ],
      "query": "Is Node Echo within the gateway's coverage range?",
      "query_id": "hard_type1_fallacy_trap",
      "type": "type1"
    },
    "expected": {
      "answer": "Uncertain",
      "premises_used": [
        0,
        1,
        2
      ]
    }
  },
  {
    "request_payload": {
      "options": [],
      "premises": [
        "If a deep learning model is trained on the ImageNet-X dataset, it achieves a baseline accuracy of 84 percent.",
        "Whenever a model achieves a baseline accuracy of 84 percent, its initial inference latency is exactly 45 milliseconds.",
        "If a model's initial inference latency is 45 milliseconds and it successfully undergoes FP16 quantization, its latency drops to 12 milliseconds.",
        "Model Vision-B is trained on the ImageNet-X dataset.",
        "Model Vision-B failed to undergo FP16 quantization.",
        "Model Vision-B occupies 250 megabytes of memory."
      ],
      "query": "What is the inference latency of Model Vision-B in milliseconds?",
      "query_id": "hard_type1_nested_numeric",
      "type": "type1"
    },
    "expected": {
      "answer": "45",
      "premises_used": [
        0,
        1,
        3
      ]
    }
  },
  {
    "request_payload": {
      "options": [],
      "premises": [
        "Every software engineer who architected microservices must master gRPC protocols.",
        "Any developer who masters gRPC protocols and holds a security clearance is assigned to Project Titan.",
        "Project Titan currently involves exactly 5 core modules.",
        "Alice masters gRPC protocols but lacks a security clearance.",
        "Bob architected microservices and holds a security clearance.",
        "Charlie holds a security clearance but has never worked on microservices."
      ],
      "query": "Which developer is assigned to Project Titan?",
      "query_id": "hard_type1_text_extraction",
      "type": "type1"
    },
    "expected": {
      "answer": "Bob",
      "premises_used": [
        0,
        1,
        4
      ]
    }
  },

    # ==========================
    # TYPE 2: PHYSICS (Physics & Scholarship)
    # ==========================
    {
        "name": "Type 2 - Physics Calculation (Kinematics)",
        "payload": {
            "query_type": "type2",
            "question": "A car travels 100 km in 2 hours. What is its average speed in km/h?\nA. 30\nB. 40\nC. 50\nD. 60",
            "options": ["A", "B", "C", "D"]
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
            "question": "What is the formula for calculating the kinetic energy of an object given its mass and velocity?\nA. E = mc^2\nB. K = 1/2 m v^2\nC. F = ma\nD. P = mv",
            "options": ["A", "B", "C", "D"]
        },
        "expected_answer": "B"
    }
]

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

# Append exact_pipeline root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from exact_pipeline.orchestration.pipeline import ExactPipeline, infer_query_type

def print_detailed_solving_process(idx, total, name, payload, expected, result, duration, is_pass):
    print("\n" + "="*90)
    status_emoji = "✅ PASS" if is_pass else "❌ FAIL"
    print(f"[{idx}/{total}] RUNNING: {name} | {status_emoji} (⏱️ {duration:.2f}s)")
    print("="*90)
    
    # 1. INPUT PARAMETERS
    print("\n📥 [PHASE 1: INPUT DATA]")
    print(f"   ❓ Question: {payload.get('question')}")
    premises_nl = payload.get('premises-NL', payload.get('premises', []))
    if premises_nl:
        print("   📜 Natural Language Premises:")
        for i, p in enumerate(premises_nl, 1):
            print(f"      P{i}: {p}")
    premises_fol = payload.get('premises-FOL', payload.get('fol', []))
    if premises_fol:
        print("   🧬 First-Order Logic (FOL) Premises (from payload):")
        for i, f in enumerate(premises_fol, 1):
            print(f"      F{i}: {f}")
    options = payload.get('options', [])
    if options:
        print(f"   🔠 Options: {options}")
        
    # 2. ROUTING & CLASSIFICATION
    q_type = infer_query_type(payload)
    route_info = result.metadata.get("route_info", {}) if result else {}
    print("\n🗺️ [PHASE 2: ROUTING & CLASSIFICATION]")
    print(f"   🔹 Inferred Task Type: {q_type.upper()} ({'Logic' if q_type == 'type1' else 'Physics'})")
    if route_info:
        print(f"   🔹 Route Info: {json.dumps(route_info)}")
    print(f"   🔹 Solver Source: {result.source if result else 'N/A'}")

    # 3. ACTUAL ROUTING RESULT. Do not rebuild a fresh Horn state here: doing
    # so made the report look as if Horn executed even when the request had
    # already bypassed it and gone directly to the typed symbolic compiler.
    if q_type == "type1":
        print("\n⚙️ [PHASE 3: LOGIC EXECUTOR ROUTING]")
        metadata = result.metadata if result else {}
        executor = metadata.get("executor", "unknown")
        bypass_reasons = metadata.get("horn_bypass_reasons", [])
        print(f"   🔹 Actual executor: {executor}")
        if executor == "deterministic-horn":
            print("   🟢 Horn fast path produced the returned proof.")
        elif bypass_reasons:
            print(f"   🟡 Horn was not executed. Bypass reasons: {bypass_reasons}")
        else:
            print(f"   🔹 Returned source: {result.source if result else 'N/A'}")
    else:
        # Physics Cache Check
        print("\n⚙️ [PHASE 3: PHYSICS CACHE / FAST PATH]")
        if result and result.source == "fast_cache":
            print(f"   🟢 Cache Hit: Matched templated variant in code_cache")
        elif result and result.source == "fast-path-sympy":
            print(f"   🟢 Fast Path SymPy: Evaluated formula directly")
        else:
            print(f"   🟡 Cache Bypass: Cache miss or fast-path not triggered")

    # 4. KNOWLEDGE RETRIEVAL & RAG
    if result and result.metadata:
        retrieval_score = result.metadata.get("retrieval_score")
        if retrieval_score is not None:
            print("\n📚 [PHASE 4: KNOWLEDGE RETRIEVAL & RAG]")
            print(f"   🔹 Retrieval Score: {retrieval_score}")
            if result.matched_id:
                print(f"   🔹 Matched Problem ID: {result.matched_id}")
        
    # 5. SEMANTIC ANALYSIS & GLOSSARY
    if q_type == "type1" and result and result.metadata:
        glossary = result.metadata.get("global_glossary")
        semantics = result.metadata.get("semantics")
        if glossary or semantics:
            print("\n🔮 [PHASE 5: SEMANTIC ANALYSIS & GLOSSARY]")
            if glossary:
                print("   🔹 Extracted Glossary Mapping:")
                for k, v in glossary.items():
                    print(f"      - {k} -> {v}")
            if semantics:
                print(f"   🔹 Extracted Target Semantics: {json.dumps(semantics)}")

    # 6. FOL TRANSLATION
    if q_type == "type1" and result and result.fol:
        print("\n🧬 [PHASE 6: FIRST-ORDER LOGIC (FOL) TRANSLATION]")
        print("   🔹 Premises & Question Translated to FOL:")
        print(f"      {result.fol.replace(chr(10), chr(10) + '      ')}")

    # 7. CODE EXECUTION (Z3 / PYTHON SANDBOX)
    if result and result.metadata:
        executor = result.metadata.get("executor")
        executed_code = result.metadata.get("executed_code")
        stdout = result.metadata.get("stdout")
        errors = result.metadata.get("execution_errors")
        
        if executor or executed_code:
            print(f"\n💻 [PHASE 7: CODE SANDBOX EXECUTION ({executor.upper() if executor else 'N/A'})]")
            if executed_code:
                print("   🔹 Generated/Executed Code:")
                print("      " + "-"*40)
                print(f"      {executed_code.strip().replace(chr(10), chr(10) + '      ')}")
                print("      " + "-"*40)
            if stdout:
                print(f"   🔹 Execution Output (Stdout): {stdout.strip()}")
            if errors:
                print(f"   ⚠️ Execution Errors: {errors}")

    # 8. FINAL OUTPUT & NORMALIZATION
    if result:
        print("\n🏁 [PHASE 8: FINAL ANSWER & NORMALIZATION]")
        print(f"   🔹 Inferred Answer: '{result.answer}'")
        if result.unit:
            print(f"   🔹 Unit: '{result.unit}'")
        if result.premises_used:
            print(f"   🔹 Premises Used Indices: {result.premises_used}")
            # Map indices to actual premise texts
            for idx_p in result.premises_used:
                if 0 <= idx_p < len(premises_nl):
                    print(f"      - P{idx_p + 1}: {premises_nl[idx_p]}")
        print(f"   🔹 Explanation: {result.explanation}")
        if result.cot:
            print("   🔹 Steps (CoT):")
            for step in result.cot:
                print(f"      - {step}")

    print("\n" + "-"*90)

def run_tests():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n==================================================")
    print("🚀 EXACT PIPELINE DIVERSE TEST SUITE (LOCAL DEBUG MODE)")
    print("==================================================\n")
    
    print("[INIT] Initializing Pipeline...")
    t0 = time.time()
    pipeline = ExactPipeline()
    print(f"[INIT] Done in {time.time()-t0:.2f}s\n")
    
    passed = 0
    total = len(TESTCASES)
    results = []

    for idx, test in enumerate(TESTCASES, 1):
        if "request_payload" in test:
            raw_payload = test["request_payload"]
            name = raw_payload.get("query_id", f"Test {idx}")
            payload = {
                "query_type": raw_payload.get("type", raw_payload.get("query_type", "type1")),
                "premises-NL": raw_payload.get("premises", raw_payload.get("premises-NL", [])),
                "question": raw_payload.get("query", raw_payload.get("question", "")),
                "options": raw_payload.get("options", [])
            }
            expected = test.get("expected", {}).get("answer", "N/A")
        else:
            payload = test["payload"]
            name = test.get("name", f"Test {idx}")
            expected = test.get("expected_answer", "N/A")
        
        t0 = time.time()
        try:
            result = pipeline.answer_result(payload)
            answer = str(result.answer).strip()
            duration = time.time() - t0
            
            # Check if answer contains expected (simple substring match for robustness)
            # If expected is empty, any non-error response passes
            if not expected:
                is_pass = True
            else:
                if len(expected.strip()) == 1:
                    is_pass = expected.strip().lower() == answer.strip().lower()
                else:
                    is_pass = expected.lower() in answer.lower()
                
            if is_pass:
                passed += 1
                
            print_detailed_solving_process(idx, total, name, payload, expected, result, duration, is_pass)
                    
            results.append({
                "name": name,
                "status": "✅ PASS" if is_pass else "❌ FAIL",
                "duration": duration,
                "expected": expected,
                "got": answer
            })
            
        except Exception as e:
            duration = time.time() - t0
            import traceback
            traceback.print_exc()
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
