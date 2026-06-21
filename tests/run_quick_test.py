import sys
import time
import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("EXACT_LLM_BASE_URL", "http://localhost:8001")
os.environ.setdefault("EXACT_LLM_MODEL", "Qwen3-8B-Instruct")

# Append exact_pipeline root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from exact_pipeline.orchestration.pipeline import ExactPipeline
from exact_pipeline.engines.horn_reasoner import (
    try_deterministic_logic,
    build_reasoning_state,
    _close_under_rules,
)
from exact_pipeline.engines.logic import _infer_symbolic_intent, classify_logic_query_type

def run_quick_tests():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n==================================================")
    print("🚀 EXACT PIPELINE QUICK TEST SUITE (LOCAL DEBUG MODE)")
    print("==================================================\n")
    
    test_json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "exact_eval_round1_logic.json"))
    with open(test_json_path, "r") as f:
        testcases = json.load(f)
        
    print("[INIT] Initializing Pipeline...")
    t0 = time.time()
    pipeline = ExactPipeline()
    print(f"[INIT] Done in {time.time()-t0:.2f}s\n")
    
    passed = 0
    total = len(testcases)
    
    for idx, test in enumerate(testcases, 1):
        try:
            if "request_payload" in test:
                payload = test["request_payload"]
                expected = test.get("expected", {}).get("answer", "")
            else:
                payload = test
                expected = test.get("expected", {}).get("answer", test.get("answer", ""))
                
            query_id = payload.get("query_id", f"Test {idx}")
            question_text = payload.get("query", payload.get("question", "No question text"))
            
            print("\n" + "="*80)
            print(f"[{idx}/{total}] Running: {query_id}")
            print(f"    📝 Question: {str(question_text).split(chr(10))[0][:80]}...")
            print("="*80)
            
            # Format payload for PipelineRequest
            api_payload = {
                "query_type": payload.get("type", payload.get("query_type", "type1")),
                "premises-NL": payload.get("premises", []),
                "question": question_text,
                "options": payload.get("options", [])
            }
            
            # --- HORN REASONER DEBUG SECTION ---
            query_type_raw = payload.get("type", payload.get("query_type", "type1"))
            if query_type_raw in ("type1", "logic", "logic_based", "logic based"):
                print("\n    🔍 [DEBUG HORN REASONER]")
                premises_nl = payload.get("premises", payload.get("premises-NL", []))
                premises_fol = payload.get("fol", payload.get("premises-FOL", []))
                options = payload.get("options", [])
                
                # 1. Parse and build initial state
                state = build_reasoning_state(premises_nl, premises_fol)
                
                print(f"      🔹 Ingested Facts ({len(state.facts)}):")
                for fact, support in state.facts.items():
                    entity_lbl = state.entity_labels.get(fact.entity, fact.entity)
                    print(f"        - {fact.name}({entity_lbl}) | Negated: {fact.negated} | (from premise indices: {[s + 1 for s in support]})")
                
                print(f"      🔹 Ingested Rules ({len(state.rules)}):")
                for rule in state.rules:
                    ants = ", ".join(f"{a.name}({state.entity_labels.get(a.entity, a.entity)})" for a in rule.antecedents)
                    print(f"        - {ants} -> {rule.consequent.name}({state.entity_labels.get(rule.consequent.entity, rule.consequent.entity)}) (premise index: {rule.premise_index + 1})")
                    
                if state.numeric_facts:
                    print(f"      🔹 Ingested Numeric Facts ({len(state.numeric_facts)}):")
                    for nf in state.numeric_facts:
                        print(f"        - {nf.subject_label} has {nf.value} {nf.measure_key} (premise index: {nf.premise_index + 1})")
                
                # 2. Run rule closure (forward chaining)
                _close_under_rules(state)
                print(f"      🔹 Facts after Forward Chaining ({len(state.facts)}):")
                for fact, support in state.facts.items():
                    entity_lbl = state.entity_labels.get(fact.entity, fact.entity)
                    print(f"        - {fact.name}({entity_lbl}) | Negated: {fact.negated} | (supported by premise indices: {[s + 1 for s in support]})")
                
                # 3. Classify and evaluate
                q_type = classify_logic_query_type(options)
                intent = _infer_symbolic_intent(question_text, q_type)
                
                print(f"      🔹 Query Classification: {q_type} | Intent: {intent}")
                
                det_res = try_deterministic_logic(
                    question=question_text,
                    premises_nl=premises_nl,
                    premises_fol=premises_fol,
                    options=options,
                    query_type=q_type,
                    intent=intent
                )
                if det_res:
                    print(f"      🟢 Horn Reasoner SUCCESS! Output generated deterministically:")
                    print(f"        - Answer: {det_res.answer}")
                    print(f"        - Explanation: {det_res.explanation}")
                    print(f"        - Premises Used Indices: {[p + 1 for p in det_res.premises_used] if det_res.premises_used else []}")
                else:
                    print(f"      🟡 Horn Reasoner returned None (could not solve, will fallback to Z3/LLM).")
                print("    " + "-"*40 + "\n")

            t0 = time.time()
            result = pipeline.answer_result(api_payload)
            result_data = {
                "answer": result.answer,
                "explanation": result.explanation,
                "fol": result.fol,
                "metadata": result.metadata
            }

                
            answer = str(result_data.get('answer', '')).strip()
            expected = str(expected).strip()
            duration = time.time() - t0
            
            # Compare
            passed_test = False
            if expected.lower() == answer.lower():
                passed_test = True
            else:
                # Try float comparison
                import re
                def extract_float(s):
                    # Handle x * 10^y format and plain floats
                    try:
                        s = str(s).replace('×', '*').replace(' ', '')
                        if '*10^' in s:
                            base, exp = s.split('*10^')
                            return float(base) * (10 ** float(exp))
                        return float(re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s).group())
                    except:
                        return None
                        
                f_exp = extract_float(expected)
                f_ans = extract_float(answer)
                if f_exp is not None and f_ans is not None:
                    import math
                    if math.isclose(f_exp, f_ans, rel_tol=1e-2):
                        passed_test = True

            if passed_test:
                passed += 1
                print(f"\n    ✅ PASS (⏱️ {duration:.2f}s) | Expected: {expected} | Got: {answer}")
            else:
                print(f"\n    ❌ FAIL (⏱️ {duration:.2f}s) | Expected: {expected} | Got: {answer}")
                print(f"    🔍 Explanation: {result_data.get('explanation', 'No explanation')}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    ⚠️ ERROR: {e}")
            
    print("\n==================================================")
    print(f"📊 TEST SUMMARY: {passed}/{total} PASSED")
    print("==================================================")

if __name__ == "__main__":
    run_quick_tests()
