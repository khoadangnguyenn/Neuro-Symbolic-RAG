from typing import Dict, List, Optional, Tuple
import re
import time
import z3
from exact_pipeline.engines.fol_parser import AdvancedZ3Transformer, sota_parse_text

def check_entailment(
    premises_dict: Dict[str, z3.ExprRef], target_expr: z3.ExprRef, ctx: z3.Context,
    timeout_ms: int = 1000,
) -> Tuple[str, List[str]]:
    solver = z3.Solver(ctx=ctx)
    solver.set("timeout", max(1, timeout_ms))
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


def check_consistency(
    premises_dict: Dict[str, z3.ExprRef], ctx: z3.Context, timeout_ms: int = 1000
) -> Tuple[Optional[bool], List[str]]:
    solver = z3.Solver(ctx=ctx)
    solver.set("timeout", max(1, timeout_ms))
    tracked = {}
    for premise_id, expression in premises_dict.items():
        marker = z3.Bool(f"consistency_{premise_id}", ctx=ctx)
        solver.assert_and_track(expression, marker)
        tracked[marker] = premise_id
    status = solver.check()
    if status == z3.sat:
        return True, []
    if status == z3.unsat:
        return False, [tracked[item] for item in solver.unsat_core() if item in tracked]
    return None, []


def check_implication(
    antecedent: z3.ExprRef,
    consequent: z3.ExprRef,
    ctx: z3.Context,
    timeout_ms: int = 1000,
) -> Optional[bool]:
    """Return whether ``antecedent`` logically entails ``consequent``.

    This compares option meanings themselves.  Unsat-core inclusion is not a
    logical strength relation and therefore must not be used for "strongest
    conclusion" questions.
    """

    solver = z3.Solver(ctx=ctx)
    solver.set("timeout", max(1, timeout_ms))
    status = solver.check(z3.And(antecedent, z3.Not(consequent)))
    if status == z3.unsat:
        return True
    if status == z3.sat:
        return False
    return None


class LogicDispatcher:
    def __init__(self, ctx: z3.Context, deadline: Optional[float] = None):
        self.ctx = ctx
        self.deadline = deadline

    def _timeout_ms(self) -> int:
        if self.deadline is None:
            return 1000
        remaining_ms = int((self.deadline - time.monotonic()) * 1000)
        return max(1, min(2500, remaining_ms))

    def dispatch(
        self,
        query_type: str,
        intent: str,
        premises_dict: Dict[str, z3.ExprRef],
        options_dict: Dict[str, z3.ExprRef],
        target_expr: Optional[z3.ExprRef] = None,
        premises_consistent: Optional[bool] = True,
        inconsistency_option: Optional[str] = None,
        inconsistency_core: Optional[List[str]] = None,
        answer_const: Optional[z3.ExprRef] = None,
        answer_candidates: Optional[Dict[str, str]] = None,
    ) -> dict:

        if premises_consistent is None:
            return {
                "verdict": "Uncertain",
                "premises_used": [],
                "explanation": "Z3 returned unknown while checking premise consistency within its resource limit.",
            }

        if not premises_consistent:
            if query_type == "multiple_choice" and inconsistency_option:
                return {
                    "verdict": "True",
                    "best_option": inconsistency_option,
                    "premises_used": list(inconsistency_core or []),
                    "explanation": "The premise theory is inconsistent (UNSAT).",
                }
            return {
                "verdict": "Inconsistent",
                "premises_used": list(inconsistency_core or []),
                "explanation": "The premise theory is inconsistent; both a target and its negation are classically entailed.",
            }

        if query_type == "open_ended":
            if target_expr is None or answer_const is None:
                return {
                    "verdict": "Uncertain",
                    "premises_used": [],
                    "explanation": "No valid symbolic answer query was provided.",
                }
            supported = []
            for candidate_id, label in (answer_candidates or {}).items():
                candidate = z3.Const(candidate_id, answer_const.sort())
                instantiated = z3.substitute(target_expr, (answer_const, candidate))
                verdict, used = check_entailment(
                    premises_dict, instantiated, self.ctx, self._timeout_ms()
                )
                if verdict == "True":
                    supported.append((candidate_id, label, used))
            if not supported:
                return {
                    "verdict": "Uncertain",
                    "premises_used": [],
                    "explanation": "No declared answer candidate is entailed by the premises.",
                }
            answers = []
            units = []
            used_premises = set()
            for _, label, used in supported:
                answer, unit = _format_answer_value(label)
                if answer not in answers:
                    answers.append(answer)
                if unit and unit not in units:
                    units.append(unit)
                used_premises.update(used)
            return {
                "verdict": "Answer",
                "answer": ", ".join(answers),
                "unit": units[0] if len(units) == 1 else "",
                "premises_used": sorted(used_premises),
                "explanation": (
                    "The symbolic query is entailed for answer candidate"
                    + ("s " if len(answers) != 1 else " ")
                    + ", ".join(answers)
                    + "."
                ),
            }

        if query_type == "yes_no_uncertain":
            if target_expr is None:
                return {
                    "verdict": "Uncertain",
                    "explanation": "No valid target provided.",
                    "premises_used": [],
                }

            verdict, used = check_entailment(
                premises_dict, target_expr, self.ctx, self._timeout_ms()
            )
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
            if intent == "verify_true":
                intent = "choose_true"
            elif intent == "verify_false":
                intent = "choose_false"
            valid_candidates = []

            for opt_id, opt_expr in options_dict.items():
                verdict, used = check_entailment(
                    premises_dict, opt_expr, self.ctx, self._timeout_ms()
                )

                if intent in ["choose_true", "choose_strongest_conclusion", "choose_fewest_premises", "open_ended"] and verdict == "True":
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
                smallest = min(len(candidate[2]) for candidate in valid_candidates)
                best_candidates = [
                    candidate for candidate in valid_candidates
                    if len(candidate[2]) == smallest
                ]
                if len(best_candidates) != 1:
                    return {
                        "verdict": "Uncertain",
                        "premises_used": [],
                        "explanation": (
                            "The minimum-premise criterion does not identify a unique option: "
                            + ", ".join(candidate[0] for candidate in best_candidates)
                            + "."
                        ),
                    }
                best = best_candidates[0]
                return {
                    "verdict": "True",
                    "best_option": best[0],
                    "premises_used": best[2],
                    "explanation": f"{best[0]} requires least number of premises.",
                }

            if intent == "choose_strongest_conclusion":
                strongest = []
                comparison_unknown = False
                for candidate in valid_candidates:
                    entails_all = True
                    for other in valid_candidates:
                        if candidate[0] == other[0]:
                            continue
                        implication = check_implication(
                            candidate[1], other[1], self.ctx, self._timeout_ms()
                        )
                        if implication is None:
                            comparison_unknown = True
                            entails_all = False
                            break
                        if not implication:
                            entails_all = False
                            break
                    if entails_all:
                        strongest.append(candidate)
                if len(strongest) != 1:
                    detail = "Z3 returned unknown for at least one strength comparison" if comparison_unknown else (
                        "no unique option logically entails all other supported options"
                    )
                    return {
                        "verdict": "Uncertain",
                        "premises_used": [],
                        "explanation": f"The strongest conclusion is ambiguous: {detail}.",
                    }
                best = strongest[0]
                return {
                    "verdict": "True",
                    "best_option": best[0],
                    "premises_used": best[2],
                    "explanation": f"{best[0]} is the strongest conclusion.",
                }

            return {
                "verdict": "Uncertain",
                "premises_used": [],
                "explanation": (
                    "More than one option satisfies the requested truth condition: "
                    + ", ".join(candidate[0] for candidate in valid_candidates)
                    + "."
                ),
            }

        if target_expr is not None:
            verdict, used = check_entailment(
                premises_dict, target_expr, self.ctx, self._timeout_ms()
            )
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


def _format_answer_value(label: str) -> Tuple[str, str]:
    """Split a leading numeric literal from its unit; preserve named entities."""

    text = str(label or "").strip()
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s+(.+)", text)
    if match:
        return match.group(1), match.group(2).strip()
    return text, ""


def run_symbolic_solver(semantics: dict, translation: dict) -> dict:
    if isinstance(translation, list):
        if len(translation) > 0 and isinstance(translation[0], dict):
            translation = translation[0]
        else:
            translation = {}
            
    if not isinstance(translation, dict):
        translation = {}

    function_types = {}

    for pred in translation.get("predicates", []):
        clean_name = pred.split("(")[0].strip()
        function_types[clean_name] = "Bool"

    for func in translation.get("functions", []):
        if ":" in func:
            raw_name, t = func.split(":")
            clean_name = raw_name.split("(")[0].strip()
            function_types[clean_name] = t.strip()
        else:
            clean_name = func.split("(")[0].strip()
            function_types[clean_name] = "Int"

    all_formula_text = []
    for key in ("premises_fol", "options_fol"):
        all_formula_text.extend(str(item) for item in translation.get(key, []))
    for key in ("condition_fol", "target_fol"):
        value = translation.get(key, "")
        if value:
            all_formula_text.append(str(value))
    inferred_numeric_types = {}
    _infer_numeric_types(all_formula_text, inferred_numeric_types)
    called_symbols = {
        name
        for raw in all_formula_text
        for name in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", _normalize_solver_fol(raw))
    }
    constant_types = {}
    for name, inferred_type in inferred_numeric_types.items():
        if name in called_symbols:
            function_types[name] = inferred_type
        else:
            constant_types[name] = inferred_type

    ctx = z3.Context()
    transformer = AdvancedZ3Transformer(
        types_dict=function_types,
        constant_types=constant_types,
        context=ctx,
    )
    solver_deadline = time.monotonic() + 7.0
    requested_deadline = semantics.get("deadline_monotonic")
    if isinstance(requested_deadline, (int, float)):
        # Keep a small return/normalization margin inside the API's 60-second
        # limit while preserving the standalone solver's normal 7s budget.
        solver_deadline = min(solver_deadline, float(requested_deadline) - 0.25)
    dispatcher = LogicDispatcher(ctx=ctx, deadline=solver_deadline)

    premises_dict = {}
    options_dict = {}
    target_expr = None
    inconsistency_option = None

    try:
        for i, fol in enumerate(translation.get("premises_fol", [])):
            clean_fol = _normalize_solver_fol(fol)
            if clean_fol:
                ast = sota_parse_text(clean_fol)
                expr = transformer.transform(ast)
                premises_dict[f"P{i+1}"] = expr

        condition_fol = translation.get("condition_fol", "")
        clean_condition = _normalize_solver_fol(condition_fol)
        if clean_condition:
            ast = sota_parse_text(clean_condition)
            expr = transformer.transform(ast)
            premises_dict["C1"] = expr

        consistency_timeout = max(
            1, min(1000, int((solver_deadline - time.monotonic()) * 1000))
        )
        premises_consistent, inconsistency_core = check_consistency(
            premises_dict, ctx, consistency_timeout
        )

        target_fol = translation.get("target_fol", "")
        if target_fol and str(target_fol).strip():
            clean_target = _normalize_solver_fol(re.sub(r"^[A-Z]\.\s*", "", str(target_fol).strip()))
            if not clean_target:
                target_expr = None
            else:
                ast = sota_parse_text(clean_target)
                target_expr = transformer.transform(ast)

        for i, fol in enumerate(translation.get("options_fol", [])):
            clean_fol = _normalize_solver_fol(re.sub(r"^[A-Z]\.\s*", "", str(fol).strip()))
            if clean_fol:
                option_id = chr(65+i)
                if clean_fol == "__theory_inconsistent__":
                    inconsistency_option = option_id
                    continue
                ast = sota_parse_text(clean_fol)
                expr = transformer.transform(ast)
                # Ensure options are matched (A, B, C, D) if provided, or O1, O2...
                # Try to map A, B, C, D from the intent if possible, else 0-indexed
                options_dict[option_id] = expr 

        answer_const = transformer.constants.get("answer")
        raw_candidate_labels = translation.get("constant_labels", {})
        answer_candidates = {
            str(candidate): str(label)
            for candidate, label in raw_candidate_labels.items()
            if isinstance(candidate, str) and isinstance(label, str)
            and candidate != "answer"
        } if isinstance(raw_candidate_labels, dict) else {}

        return dispatcher.dispatch(
            query_type=semantics.get("query_type", "open_ended"),
            intent=semantics.get("intent", "open_analysis"),
            premises_dict=premises_dict,
            options_dict=options_dict,
            target_expr=target_expr,
            premises_consistent=premises_consistent,
            inconsistency_option=inconsistency_option,
            inconsistency_core=inconsistency_core,
            answer_const=answer_const,
            answer_candidates=answer_candidates,
        )

    except Exception as e:
        return {
            "verdict": "Error",
            "explanation": f"Solver exception: {str(e)}",
            "premises_used": [],
        }


def _normalize_solver_fol(fol: object) -> str:
    text = str(fol or "").strip()
    if not text or text.startswith("/*"):
        return ""
    text = text.replace("≥", ">=").replace("≤", "<=").replace("≠", "!=")
    text = re.sub(
        r"(?<![A-Za-z0-9_])['\"]([A-Za-z][A-Za-z0-9 _-]*)['\"]",
        lambda match: re.sub(r"\W+", "_", match.group(1)).strip("_"),
        text,
    )
    def normalize_negative_predicate(match: re.Match) -> str:
        prefix, stem, arguments = match.groups()
        if prefix == "lacks_":
            stem = "has_" + stem
        elif prefix == "fails_":
            stem = "passes_" + stem
        elif stem.startswith("have_"):
            stem = "has_" + stem[len("have_"):]
        return f"¬{stem}({arguments})"

    text = re.sub(
        r"\b(does_not_|do_not_|did_not_|cannot_|can_not_|not_|lacks_|fails_)"
        r"([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)",
        normalize_negative_predicate,
        text,
    )
    text = re.sub(r"(?<![\w:])(\d{1,2}):(\d{2})(?![\w:])", r"time_\1_\2", text)
    text = _normalize_function_arguments(text)
    return text


def _normalize_function_arguments(fol: str) -> str:
    def repl(match: re.Match) -> str:
        name = match.group(1)
        raw_args = match.group(2)
        if any(sym in raw_args for sym in ("→", "->", "∧", "&", "∨", "|", "=", "<", ">")):
            return match.group(0)
        args = []
        for arg in raw_args.split(","):
            clean = arg.strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*-[A-Za-z0-9_-]+", clean):
                clean = re.sub(r"\W+", "_", clean).strip("_")
            elif re.search(r"\s", clean) and not re.search(r"\s[-+*/]\s", clean):
                clean = re.sub(r"\W+", "_", clean).strip("_")
            args.append(clean)
        return f"{name}({', '.join(args)})"

    return re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)", repl, fol)


def _infer_numeric_types(formulas: List[str], types_config: Dict[str, str]) -> None:
    number = r"[-+]?\d+(?:\.\d+)?"
    comp = r"(?:=|==|!=|<=|>=|<|>)"
    for raw in formulas:
        text = _normalize_solver_fol(raw)
        if not text:
            continue
        for name in re.findall(rf"\b([A-Za-z_][A-Za-z0-9_]*)\([^)]*\)\s*{comp}\s*{number}", text):
            types_config[name] = "Int"
        for name in re.findall(rf"\b([A-Za-z_][A-Za-z0-9_]*)\s*{comp}\s*{number}", text):
            if name not in {"x", "y", "z"}:
                types_config[name] = "Int"
        for name in re.findall(rf"{number}\s*{comp}\s*\b([A-Za-z_][A-Za-z0-9_]*)\b", text):
            if name not in {"x", "y", "z"}:
                types_config[name] = "Int"
