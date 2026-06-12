from typing import Dict, List, Optional, Tuple
import z3
from exact_pipeline.engines.fol_parser import AdvancedZ3Transformer, sota_parse_text

def check_entailment(
    premises_dict: Dict[str, z3.ExprRef], target_expr: z3.ExprRef, ctx: z3.Context
) -> Tuple[str, List[str]]:
    solver = z3.Solver(ctx=ctx)
    solver.set("timeout", 2000)
    solver.set("smt.core.minimize", True)

    track_map = {}
    for p_id, expr in premises_dict.items():
        track_var = z3.Bool(f"track_{p_id}", ctx=ctx)
        solver.assert_and_track(expr, track_var)
        track_map[track_var] = p_id

    if solver.check(z3.Not(target_expr)) == z3.unsat:
        core = solver.unsat_core()
        used_premises = [track_map[var] for var in core if var in track_map]
        return "True", used_premises

    if solver.check(target_expr) == z3.unsat:
        core = solver.unsat_core()
        used_premises = [track_map[var] for var in core if var in track_map]
        return "False", used_premises

    return "Uncertain", []


class LogicDispatcher:
    def __init__(self, ctx: z3.Context):
        self.ctx = ctx

    def dispatch(
        self,
        query_type: str,
        intent: str,
        premises_dict: Dict[str, z3.ExprRef],
        options_dict: Dict[str, z3.ExprRef],
        target_expr: Optional[z3.ExprRef] = None,
    ) -> dict:

        if query_type == "yes_no_uncertain":
            if target_expr is None:
                return {
                    "verdict": "Uncertain",
                    "explanation": "No valid target provided.",
                    "premises_used": [],
                }

            verdict, used = check_entailment(premises_dict, target_expr, self.ctx)
            is_expected = (verdict == "False" and intent == "verify_false") or (
                verdict == "True" and intent != "verify_false"
            )

            return {
                "verdict": verdict,
                "is_expected": is_expected,
                "premises_used": used,
                "explanation": f"Target evaluates to {verdict}.",
            }

        elif query_type == "multiple_choice":
            valid_candidates = []

            for opt_id, opt_expr in options_dict.items():
                verdict, used = check_entailment(premises_dict, opt_expr, self.ctx)

                if intent in ["choose_true", "choose_strongest_conclusion", "choose_fewest_premises"] and verdict == "True":
                    valid_candidates.append((opt_id, opt_expr, used))
                elif intent == "choose_false" and verdict == "False":
                    valid_candidates.append((opt_id, opt_expr, used))

            if not valid_candidates:
                return {
                    "verdict": "Uncertain",
                    "explanation": "No option satisfies conditions.",
                    "premises_used": [],
                }

            if len(valid_candidates) == 1:
                return {
                    "verdict": "True",
                    "best_option": valid_candidates[0][0],
                    "premises_used": valid_candidates[0][2],
                    "explanation": f"Option {valid_candidates[0][0]} is valid.",
                }

            if intent == "choose_fewest_premises":
                best = min(valid_candidates, key=lambda x: len(x[2]))
                return {
                    "verdict": "True",
                    "best_option": best[0],
                    "premises_used": best[2],
                    "explanation": f"{best[0]} requires least number of premises.",
                }

            if intent == "choose_strongest_conclusion":
                best = valid_candidates[0]
                for cand in valid_candidates[1:]:
                    solver = z3.Solver(ctx=self.ctx)
                    solver.set("timeout", 2000)

                    # Compare strength INDEPENDENT of the premises
                    # cand -> best (cand is stronger)
                    solver.add(cand[1])
                    solver.add(z3.Not(best[1]))

                    if solver.check() == z3.unsat:
                        best = cand

                return {
                    "verdict": "True",
                    "best_option": best[0],
                    "premises_used": best[2],
                    "explanation": f"{best[0]} is the strongest conclusion.",
                }

            return {
                "verdict": "True",
                "best_option": valid_candidates[0][0],
                "premises_used": valid_candidates[0][2],
                "explanation": "Multiple valid options exist.",
            }

        if target_expr is not None:
            verdict, used = check_entailment(premises_dict, target_expr, self.ctx)
            return {
                "verdict": verdict,
                "explanation": f"Result: {verdict}",
                "premises_used": used,
            }

        return {
            "verdict": "Uncertain",
            "explanation": "Unsupported query type.",
            "premises_used": [],
        }


def run_symbolic_solver(semantics: dict, translation: dict) -> dict:
    if isinstance(translation, list):
        if len(translation) > 0 and isinstance(translation[0], dict):
            translation = translation[0]
        else:
            translation = {}
            
    if not isinstance(translation, dict):
        translation = {}

    types_config = {}

    for pred in translation.get("predicates", []):
        clean_name = pred.split("(")[0].strip()
        types_config[clean_name] = "Bool"

    for func in translation.get("functions", []):
        if ":" in func:
            raw_name, t = func.split(":")
            clean_name = raw_name.split("(")[0].strip()
            types_config[clean_name] = t.strip()
        else:
            clean_name = func.split("(")[0].strip()
            types_config[clean_name] = "Int"

    ctx = z3.Context()
    transformer = AdvancedZ3Transformer(types_dict=types_config, context=ctx)
    dispatcher = LogicDispatcher(ctx=ctx)

    premises_dict = {}
    options_dict = {}
    target_expr = None

    try:
        for i, fol in enumerate(translation.get("premises_fol", [])):
            if fol.strip():
                ast = sota_parse_text(fol)
                expr = transformer.transform(ast)
                premises_dict[f"P{i+1}"] = expr

        condition_fol = translation.get("condition_fol", "")
        if condition_fol and condition_fol.strip() and any(c in condition_fol for c in ['∀', '∃', '¬', '∧', '∨', '→', '↔']):
            ast = sota_parse_text(condition_fol)
            expr = transformer.transform(ast)
            premises_dict["C1"] = expr

        target_fol = translation.get("target_fol", "")
        if target_fol and target_fol.strip() and any(c in target_fol for c in ['∀', '∃', '¬', '∧', '∨', '→', '↔']):
            import re
            clean_target = re.sub(r"^[A-Z]\.\s*", "", target_fol.strip())
            ast = sota_parse_text(clean_target)
            target_expr = transformer.transform(ast)

        for i, fol in enumerate(translation.get("options_fol", [])):
            if fol.strip():
                import re
                clean_fol = re.sub(r"^[A-Z]\.\s*", "", fol.strip())
                ast = sota_parse_text(clean_fol)
                expr = transformer.transform(ast)
                # Ensure options are matched (A, B, C, D) if provided, or O1, O2...
                # Try to map A, B, C, D from the intent if possible, else 0-indexed
                options_dict[chr(65+i)] = expr 

        return dispatcher.dispatch(
            query_type=semantics.get("query_type", "open_ended"),
            intent=semantics.get("intent", "open_analysis"),
            premises_dict=premises_dict,
            options_dict=options_dict,
            target_expr=target_expr,
        )

    except Exception as e:
        return {
            "verdict": "Error",
            "explanation": f"Solver exception: {str(e)}",
            "premises_used": [],
        }
