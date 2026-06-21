"""Type 1 logic-based educational QA pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence
from collections import OrderedDict
import math
import re
import json
import time
import threading

from exact_pipeline.core.data import logic_document
from exact_pipeline.engines.executors import ExecutionResult, Z3Executor
from exact_pipeline.knowledge.knowledge import get_logic_knowledge_index, merge_premises, render_logic_knowledge, get_reasoning_subgraph_context
from exact_pipeline.llm.llm import LLMError, OpenAICompatibleLLM
from exact_pipeline.core.models import LogicExample, PipelineResult
from exact_pipeline.knowledge.retrieval import SearchHit, VectorDBIndex, render_hits, CrossEncoderReranker
from exact_pipeline.utils.text_utils import first_present, normalize_key, normalize_text
from exact_pipeline.llm.templates import LOGIC_TEMPLATE
from exact_pipeline.orchestration.feedback import extract_and_write_back_logic
from exact_pipeline.knowledge.graph_db import HybridDB
from exact_pipeline.engines.horn_reasoner import (
    HornCapability,
    assess_horn_capability,
    try_deterministic_logic,
)
from exact_pipeline.engines.fol_ast import (
    AstValidationError,
    coerce_translation_envelope,
    compile_ast_translation,
    compile_compact_fol_translation,
    compile_flat_ast_translation,
    compile_prefix_ir_translation,
    logic_prefix_ir_json_schema,
)
from exact_pipeline.engines.schema_learner import PredicateSchemaLearner


# ---------------------------------------------------------------------------
# FOL POST-PROCESSING SANITIZER
# Automatically fixes common LLM translation errors in FOL formulas.
# These are PATTERN-BASED fixes that work for ANY input, not hardcoded.
# ---------------------------------------------------------------------------

def _sanitize_fol_formula(fol: str) -> str:
    """Automatically fix common FOL translation errors.

    1. Tautology detection: ∀x(P(x) → P(x)) is a tautology that provides
       no information to Z3. Convert it to the standalone fact ∀x(P(x)).
       This happens when the LLM translates 'A is true' by pattern-matching
       the 'if-then' structure of surrounding premises.

    2. Double-application: pred(x)(x) is a syntax error from the LLM
       applying a predicate twice. Fix to pred(x).

    3. Bare free-variable predicate: pred(x) without quantifier → ∀x(pred(x)).
       This happens when the LLM skips the ∀x wrapper for standalone facts.
       Without the quantifier, Z3 treats x as an unbound symbol and the
       assertion does not universally constrain the predicate, breaking chained
       inference (e.g. rains(x) ∧ ∀x(rains(x)→wet(x)) should give wet(x)).
    """
    original = fol.strip()
    result = original

    if not result or result.startswith("/*"):
        return result

    # Normalize common non-FOL literals before parsing:
    # - time values like 19:00 become constants time_19_00
    # - entity arguments with spaces become snake_case constants
    result = re.sub(r'(?<![\w:])(\d{1,2}):(\d{2})(?![\w:])', r'time_\1_\2', result)
    result = _normalize_function_arguments(result)

    # Canonicalize lexical negation into logical negation.  Treating
    # does_not_have_x as an unrelated positive predicate destroys
    # contraposition and excluded-middle reasoning in Z3.
    def normalize_negative_predicate(match: re.Match) -> str:
        prefix, stem, arguments = match.groups()
        if prefix == "lacks_":
            stem = "has_" + stem
        elif prefix == "fails_":
            stem = "passes_" + stem
        elif stem.startswith("have_"):
            stem = "has_" + stem[len("have_"):]
        return f"¬{stem}({arguments})"

    result = re.sub(
        r"\b(does_not_|do_not_|did_not_|cannot_|can_not_|not_|lacks_|fails_)"
        r"([A-Za-z_][A-Za-z0-9_]*)\(([^()]*)\)",
        normalize_negative_predicate,
        result,
    )

    # --- Fix 1: Tautology detection ---
    # Matches: ∀x(LHS → RHS) or ∃x(LHS → RHS) where LHS == RHS
    # Also matches with space: ∀x (LHS → RHS)
    tautology_re = re.compile(
        r'^([∀∃])(\w+)\s*\(\s*(.+?)\s*→\s*(.+?)\s*\)$'
    )
    m = tautology_re.match(result)
    if m:
        quantifier, var, lhs, rhs = m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()
        if lhs == rhs:
            # Tautology! Convert ∀x(P(x) → P(x)) to ∀x(P(x))
            result = f"{quantifier}{var}({lhs})"

    # Also handle arrow written as ->
    tautology_re2 = re.compile(
        r'^([∀∃])(\w+)\s*\(\s*(.+?)\s*->\s*(.+?)\s*\)$'
    )
    m2 = tautology_re2.match(result)
    if m2:
        quantifier, var, lhs, rhs = m2.group(1), m2.group(2), m2.group(3).strip(), m2.group(4).strip()
        if lhs == rhs:
            result = f"{quantifier}{var}({lhs})"

    # --- Fix 2: Double-application ---
    # Matches: predicate_name(args)(more_args) → predicate_name(args)
    # e.g. ground_gets_wet(x)(x) → ground_gets_wet(x)
    double_app_re = re.compile(r'(\b[a-zA-Z_]\w*\([^)]*\))\([^)]*\)')
    result = double_app_re.sub(r'\1', result)

    # --- Fix 3: Bare free-variable predicate without quantifier ---
    # A formula like "rains(x)" or "a(x)" at the top level has x as a free
    # variable. In Z3 this is NOT equivalent to ∀x rains(x). The sanitizer
    # wraps it: "rains(x)" → "∀x(rains(x))".
    #
    # Safety conditions before wrapping:
    #   - Formula does NOT already start with a quantifier (∀ or ∃)
    #   - Formula does NOT contain implication/connectives (→, ↔, ∧, ∨)
    #     at the top level (those are handled by their own quantifiers)
    #   - The single argument is exactly a single lowercase letter (common
    #     FOL variable convention: x, y, z) indicating a free variable.
    #   - The formula is a single predicate call (no nested predicates)
    if not re.match(r'^[∀∃]', result):
        # Match: optional negation, then pred_name(single_var)
        # where single_var is a single letter (x, y, z, etc.)
        bare_pred_re = re.compile(
            r'^(¬?)([a-zA-Z_]\w*)\(([a-z])\)$'
        )
        bm = bare_pred_re.match(result)
        if bm:
            neg, pred, var = bm.group(1), bm.group(2), bm.group(3)
            result = f"∀{var}({neg}{pred}({var}))"

    if result != original:
        print(f"   🔧 [FOL SANITIZER] '{original}' → '{result}'", flush=True)

    return result


def _normalize_function_arguments(fol: str) -> str:
    """Normalize simple entity arguments inside function calls.

    This is not a semantic rewrite; it only turns parser-hostile constants such
    as `Nova Supplies` into `Nova_Supplies`.
    """

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


def _sanitize_fol_list(fol_list: List[str]) -> List[str]:
    """Apply sanitization to a list of FOL formulas."""
    return [_sanitize_fol_formula(f) for f in fol_list]


def _validate_symbolic_translation_contract(
    translation: object,
    *,
    premise_count: int,
    query_type: str,
    option_count: int,
) -> List[str]:
    """Check legacy string-FOL completeness before invoking Z3."""
    if not isinstance(translation, dict):
        return ["translation must be an object"]

    errors: List[str] = []
    premises = translation.get("premises_fol")
    target = translation.get("target_fol")
    options = translation.get("options_fol")

    if not isinstance(premises, list) or len(premises) != premise_count:
        actual = len(premises) if isinstance(premises, list) else "non-array"
        errors.append(f"expected exactly {premise_count} premises_fol entries, got {actual}")
    elif any(not isinstance(item, str) or not item.strip() for item in premises):
        errors.append("every premises_fol entry must be a non-empty string")

    if query_type == "yes_no_uncertain":
        if not isinstance(target, str) or not target.strip():
            errors.append("yes/no queries require a non-empty target_fol")
        if options not in ([], None):
            errors.append("yes/no queries must not emit options_fol")
    elif query_type == "multiple_choice":
        if not isinstance(options, list) or len(options) != option_count:
            actual = len(options) if isinstance(options, list) else "non-array"
            errors.append(f"expected exactly {option_count} options_fol entries, got {actual}")
        elif any(not isinstance(item, str) or not item.strip() for item in options):
            errors.append("every options_fol entry must be a non-empty string")
    return errors


def _validate_compiled_semantic_alignment(
    *,
    premises_nl: Sequence[str],
    option_texts: Sequence[str],
    question: str,
    query_type: str,
    compiled: object,
) -> List[str]:
    """Check source-to-IR preservation without trying to solve the problem."""

    entity_mentions = getattr(compiled, "entity_mentions", ())
    predicate_mentions = getattr(compiled, "predicate_mentions", ())
    premise_formulas = list(getattr(compiled, "premises_fol", ()))
    option_formulas = list(getattr(compiled, "options_fol", ()))
    target_formula = str(getattr(compiled, "target_fol", ""))
    errors: List[str] = []

    def mention_matches(mention: str, source: str) -> bool:
        # Validation must not use aggressive stemming: "manufactured in"
        # and the type phrase "manufacturing node" are different concepts.
        # Require an exact normalized token sequence from the source.
        mention_text = normalize_text(mention)
        source_text = normalize_text(source)
        if not mention_text:
            return False
        return bool(re.search(
            rf"(?<![a-z0-9]){re.escape(mention_text)}(?![a-z0-9])",
            source_text,
        ))

    def audit_symbols(source: str, formula: str, label: str) -> None:
        used = set(re.findall(r"\b[ep]\d+\b", formula))
        for symbol, mentions in (*entity_mentions, *predicate_mentions):
            if any(mention_matches(mention, source) for mention in mentions) and symbol not in used:
                errors.append(f"{label} drops source concept {mentions[0]!r} ({symbol})")

    def audit_connectives(source: str, formula: str, label: str) -> None:
        text = normalize_text(source)
        if "if and only if" in text and "↔" not in formula:
            errors.append(f"{label} must preserve biconditional")
        if re.search(r"\beither\b.+\bor\b", text) and "∨" not in formula:
            errors.append(f"{label} must preserve disjunction")
        if "at least one" in text and "∃" not in formula:
            errors.append(f"{label} must preserve existential witness")
        if (
            re.search(r"\b(?:if|whenever|any|every|all)\b", text)
            and "→" not in formula
            and "↔" not in formula
        ):
            errors.append(f"{label} must preserve rule implication")
        if (
            re.search(r"\b(?:no|not|cannot|lacks?|failed|never)\b", text)
            and not re.search(r"\b(?:no\b.+\bunless|not\b.+\bunless)\b", text)
            and "¬" not in formula
        ):
            errors.append(f"{label} must preserve explicit negation")
        without_iff = text.replace("if and only if", "iff")
        conditional_antecedent = re.match(r"^if\s+(.+?)(?:,|\bthen\b)", without_iff)
        requires_conjunction = bool(
            re.search(r"\bbut\b", without_iff)
            or (
                conditional_antecedent
                and re.search(r"\band\b", conditional_antecedent.group(1))
            )
        )
        if requires_conjunction and "∧" not in formula:
            errors.append(f"{label} must preserve conjunction")

    for index, (source, formula) in enumerate(zip(premises_nl, premise_formulas)):
        audit_symbols(str(source), formula, f"premise {index}")
        audit_connectives(str(source), formula, f"premise {index}")

    normalized_formulas: Dict[str, int] = {}
    for index, formula in enumerate(premise_formulas):
        canonical = re.sub(r"\s+", "", formula)
        previous = normalized_formulas.get(canonical)
        if previous is not None and normalize_text(premises_nl[previous]) != normalize_text(premises_nl[index]):
            errors.append(f"premises {previous} and {index} collapse to the same formula")
        normalized_formulas[canonical] = index

    if query_type == "multiple_choice":
        for index, (source, formula) in enumerate(zip(option_texts, option_formulas)):
            audit_symbols(str(source), formula, f"option {index}")
            audit_connectives(str(source), formula, f"option {index}")
    else:
        audit_symbols(question, target_formula, "query target")

    return errors


def _render_symbolic_explanation(
    *,
    verdict: str,
    answer: str,
    question: str,
    query_type: str,
    option_texts: Sequence[str],
    premises_nl: Sequence[str],
    used_indices: Sequence[int],
    unit: str = "",
) -> tuple[str, dict]:
    """Render a solver-grounded explanation without generative reasoning."""

    support = [
        {"index": index, "label": f"P{index + 1}", "text": str(premises_nl[index])}
        for index in sorted(set(used_indices))
        if 0 <= index < len(premises_nl)
    ]
    cited = "; ".join(f"{item['label']}: {item['text']}" for item in support)
    conclusion = question.strip()
    method = "smt_status"
    if verdict == "Inconsistent":
        summary = "The premise theory is inconsistent (UNSAT)."
        method = "unsat_core"
    elif query_type == "multiple_choice" and answer != "Uncertain":
        option_index = ord(answer.upper()) - ord("A") if len(answer) == 1 else -1
        option = option_texts[option_index] if 0 <= option_index < len(option_texts) else answer
        conclusion = option
        summary = f"Option {answer} is entailed: {option}."
        method = "negated_option_unsat"
    elif query_type == "open_ended" and answer != "Uncertain":
        rendered_value = f"{answer} {unit}".strip()
        conclusion = rendered_value
        summary = f"The symbolic answer projection entails {rendered_value}."
        method = "candidate_substitution_entailment"
    elif verdict == "True":
        summary = "The queried statement is entailed; its negation is UNSAT with the premises."
        method = "negated_query_unsat"
    elif verdict == "False":
        summary = "The negation of the queried statement is entailed."
        method = "query_unsat"
    else:
        summary = "Neither the query nor its negation is entailed by the consistent premises."
        method = "two_sided_satisfiable"

    if cited:
        summary += " Supporting premises: " + cited
    reasoning = {
        "type": "symbolic_entailment",
        "logical_status": verdict,
        "method": method,
        "conclusion": conclusion,
        "proof_support": support,
    }
    return summary, reasoning


TRANSLATION_SYSTEM_PROMPT = """
### ROLE
You are an expert logic translator and schema extractor. Your task is a two-step process:
1. Extract a unified, consistent, and semantically deduplicated set of predicates and numerical functions from all natural language premises.
2. Convert the premises into First-Order Logic (FOL) formulas using ONLY the extracted predicates and functions.

### RULES
1. Standard FOL Symbols & Math Operations:
- Quantifiers: ∀, ∃
- Logical: ¬, ∧, ∨, →, ↔
- Mathematical comparisons:
  * "more/greater than X" -> `> X`
  * "less/fewer than X" -> `< X`
  * "at least/minimum of X" -> `>= X`
  * "at most/maximum of X" -> `<= X`
  * "exactly/equal/is X" -> `= X`
  * "not equal/other than X" -> `!= X`

2. Extraction & Naming: 
- Extract all boolean predicates (yielding True/False) and numerical functions (yielding numeric/quantifiable values).
- Format all names using strictly `snake_case` with their bound variable (e.g., `well_tested(x)`).
- For functions, explicitly state the return type in the declaration (e.g., `score(x): Int`, `gpa(x): Real`).
- Convert multi-argument relations into unary predicates/functions by embedding context into the name (e.g., "studies political theory for 12 hours" -> `study_political_theory(x) > 12`).

3. Strict Semantic Unification (Anti-Mismatch):
- Actively detect and merge phrases that share the SAME semantic meaning.
- Choose the most concise and representative name for the unified predicate.

4. Strict Vocabulary Adherence: 
- The FOL formulas MUST strictly and exclusively use the exact predicates and functions declared. Do NOT invent new names during translation.

5. Quantifier Scope & Variable Binding: 
- Every predicate/function must be associated with a variable in parentheses (e.g., P(x)). Do not use them without variables.
- Ensure the scope of all quantifiers (∀, ∃) is clearly defined using parentheses.

6. Output Format: 
Return ONLY a valid JSON object. Do not provide explanations, conversational fillers, or markdown code blocks.
The JSON must strictly follow this schema:
{
  "predicates": ["predicate_1(x)", "predicate_2(x)"],
  "functions": ["function_name(x): Type"],
  "premises_fol": ["formula1", "formula2"]
}
"""

def format_premises_for_prompt(premises: list[str]) -> str:
    if not premises:
        return ""
    return "\n".join(f"{i + 1}. {premise}" for i, premise in enumerate(premises))

def get_user_prompt(input_text: str) -> str:
    return f"""
### TASK
Translate the following statement into First-Order Logic (FOL). 
Ensure the result is accurate based on the System Rules.

### INPUT DATA

{input_text}

"""

def count_tokens(text: str) -> int:
    return len(text.split())

def split_into_elastic_batches(premises: list[str], base_threshold: int, upper_threshold: int) -> list[str]:
    sentence_tokens = [count_tokens(p) for p in premises]
    total_tokens = sum(sentence_tokens)
    
    if total_tokens == 0:
        return []
        
    min_batches = math.ceil(total_tokens / upper_threshold)
    target_batch_size = total_tokens / min_batches
    
    batches = []
    current_batch = []
    current_batch_tokens = 0
    remaining_tokens = total_tokens
    
    for sentence, tokens in zip(premises, sentence_tokens):
        remaining_tokens -= tokens
        
        exceeds_upper = (current_batch_tokens + tokens) > upper_threshold
        reached_target = current_batch_tokens >= target_batch_size
        remaining_slots = min_batches - len(batches) - 1
        safe_to_close = remaining_tokens <= (remaining_slots * upper_threshold)
        
        if (exceeds_upper or (reached_target and safe_to_close)) and current_batch:
            batches.append(" ".join(current_batch))
            current_batch = [sentence]
            current_batch_tokens = tokens
        else:
            current_batch.append(sentence)
            current_batch_tokens += tokens
            
    if current_batch:
        batches.append(" ".join(current_batch))
        
    return batches

def classify_logic_query_type(options: list[str]) -> str:
    if not options:
        return "open_ended"
    YES_NO_UNCERTAIN_SET = {"Yes", "No", "Uncertain"}
    if set(options).issubset(YES_NO_UNCERTAIN_SET):
        return "yes_no_uncertain"
    return "multiple_choice"


def _infer_symbolic_intent(question: str, query_type: str) -> str:
    """Cheap pre-LLM intent hint for the deterministic symbolic fast path."""

    text = question.lower()
    if query_type == "yes_no_uncertain":
        return "verify_false" if " not " in f" {text} " or "false" in text else "verify_true"
    if query_type == "multiple_choice":
        if "fewest" in text or "least" in text or "minimum" in text:
            return "choose_fewest_premises"
        if "strongest" in text or "most strongly" in text:
            return "choose_strongest_conclusion"
        if "false" in text or "not supported" in text or "cannot be concluded" in text:
            return "choose_false"
        return "choose_true"
    return "open_analysis"


class LogicPipeline:
    def __init__(
        self,
        examples: Sequence[LogicExample],
        *,
        db_path: str,
        graph_path: str,
        alpha: float,
        llm: Optional[OpenAICompatibleLLM] = None,
        retrieval_k: int = 5,
        high_match_threshold: float = 0.85,
        low_match_threshold: float = 0.50,
        max_retries: int = 2,
        code_timeout_s: float = 4.0,
    ) -> None:
        self.examples = list(examples)
        self.db_path = db_path
        self.graph_path = graph_path
        self.alpha = alpha
        self.llm = llm or OpenAICompatibleLLM()
        self.retrieval_k = retrieval_k
        self.high_match_threshold = high_match_threshold
        self.low_match_threshold = low_match_threshold
        self.max_retries = max(0, max_retries)
        self._symbolic_ast_cache = OrderedDict()
        self._symbolic_ast_cache_lock = threading.RLock()
        self._symbolic_ast_cache_capacity = 256
        self.z3_executor = Z3Executor(timeout_s=code_timeout_s)
        self.reranker = CrossEncoderReranker()
        schema_path = Path(graph_path).with_name("predicate_schema.json")
        self.predicate_schema = PredicateSchemaLearner.from_examples(
            self.examples,
            cache_path=schema_path,
        )

        self.index = VectorDBIndex.from_items(
            "logic_examples", 
            self.examples, 
            logic_document, 
            lambda x: x.question_id, 
            self.db_path
        )
        self.logic_knowledge_index = get_logic_knowledge_index(self.db_path, self.graph_path, self.alpha)

        self.exact_by_id: Dict[str, LogicExample] = {}
        self.exact_by_full_key: Dict[str, LogicExample] = {}
        self.exact_by_question: Dict[str, LogicExample] = {}
        
        for ex in examples:
            if ex.premises_nl and ex.question:
                key = normalize_key(ex.premises_nl, ex.question)
                self.exact_by_full_key[key] = ex
            if not ex.premises_nl and ex.question:
                self.exact_by_question[normalize_key(ex.question)] = ex

    def answer(self, payload: dict) -> PipelineResult:
        request_deadline = time.monotonic() + 55.0
        question = str(first_present(payload, ["question", "query"], "")).strip()
        premises_nl = _ensure_list(first_present(payload, ["premises-NL", "premises_nl", "premises", "context"], []))
        premises_fol = _ensure_list(first_present(payload, ["premises-FOL", "premises_fol", "fol"], []))

        if not question:
            return PipelineResult(
                answer="Uncertain",
                explanation="The request did not include a Type 1 question.",
                confidence=0.0,
                query_type="type1",
                source="validation",
            )

        options = _ensure_list(first_present(payload, ["options", "choices"], []))
        if not options:
            matches = re.findall(
                r"(?:^|\n)\s*[A-Z][\.)]\s*(.+?)(?=\n\s*[A-Z][\.)]\s*|\Z)",
                question,
                flags=re.DOTALL,
            )
            if len(matches) >= 2:
                options = [match.strip() for match in matches if match.strip()]
        query_type = classify_logic_query_type(options)
        early_intent = _infer_symbolic_intent(question, query_type)
        horn_capability = assess_horn_capability(
            question=question,
            premises_nl=premises_nl,
            premises_fol=premises_fol,
            options=options,
        )
        requires_full_symbolic = not horn_capability.supported
        deterministic_result = (
            try_deterministic_logic(
                question=question,
                premises_nl=premises_nl,
                premises_fol=premises_fol,
                options=options,
                query_type=query_type,
                intent=early_intent,
                learned_aliases=self.predicate_schema.aliases,
            )
            if horn_capability.supported
            else None
        )
        if deterministic_result is not None:
            deterministic_result.metadata["symbolic_stage"] = "pre_routing"
            deterministic_result.metadata["route_info"] = payload.get("_route_info", {})
            return deterministic_result

        # Capability detection cannot enumerate every future linguistic
        # construction.  Whenever a premise-bearing problem is not completely
        # decided by the controlled parser, route every answer shape—including
        # open value/entity projection—to the same typed AST + Z3 path.
        if deterministic_result is None and (premises_nl or premises_fol):
            requires_full_symbolic = True
            horn_capability = HornCapability(
                False,
                tuple(sorted(set(horn_capability.reasons) | {"horn_unresolved"})),
            )

        route_info = payload.get("_route_info", {})

        request_id = str(first_present(payload, ["question_id", "record_id", "id"], "")).strip()
        exact = self.exact_by_id.get(request_id) if request_id and not requires_full_symbolic else None
        if exact is not None:
            return self._from_example(exact, confidence=0.995, source="id-exact")

        full_key = normalize_key(premises_nl, question)
        exact = self.exact_by_full_key.get(full_key) if not requires_full_symbolic else None
        if exact is None and not premises_nl and not requires_full_symbolic:
            maybe = self.exact_by_question.get(normalize_key(question))
            if maybe is not None:
                exact = maybe
        if exact is not None:
            return self._from_example(exact, confidence=0.99, source="exact")
        # Query VectorDB for context using natural language.
        # Bypassing early-return here to avoid "Context Domination" 
        # (different questions sharing the same premises have near-identical vectors).
        query_text = "\n".join([*premises_nl, question])
        orchestration_data = (
            {
                "exact_entities": [],
                "semantic_anchors": [],
                "complexity_score": 5,
                "is_solvable": True,
                "intent": early_intent,
                "condition": "",
                "target": "",
                "query_type": query_type,
            }
            if requires_full_symbolic
            else self._orchestrate_query(question, query_type)
        )
        orchestration_data["query_type"] = query_type
        if requires_full_symbolic:
            orchestration_data["horn_bypass_reasons"] = list(horn_capability.reasons)
        complexity = orchestration_data.get("complexity_score", 3)
        vector_k = max(5, int(complexity * 10))
        rerank_k = max(2, int(complexity * 2))
        
        # Build expanded semantic search query
        search_query = question
        anchors = orchestration_data.get("semantic_anchors", [])
        if anchors:
            search_query += "\n" + "\n".join(anchors)
            
        hits = [] if requires_full_symbolic else self.index.search(
            search_query,
            k=vector_k,
            reranker=self.reranker,
            rerank_top_k=rerank_k,
        )

        # RAG-Based FOL Cache (Offline Pre-computation):
        # If the user provides premises_nl but NO premises_fol, we search the retrieved 
        # VectorDB hits to see if these premises were already translated during Offline Ingestion.
        if premises_nl and not premises_fol and not requires_full_symbolic:
            matched_fol = None
            
            # First, try an exact ordered match for maximum safety
            nl_tuple = tuple(premises_nl)
            for hit in hits:
                if hasattr(hit.item, "premises_nl") and tuple(hit.item.premises_nl) == nl_tuple:
                    if hasattr(hit.item, "premises_fol") and hit.item.premises_fol:
                        matched_fol = hit.item.premises_fol
                        break
                        
            # If not exact match, try mapping each premise individually from the retrieved hits
            if not matched_fol:
                hit_dict = {}
                for hit in hits:
                    if hasattr(hit.item, "premises_nl") and hasattr(hit.item, "premises_fol"):
                        if len(hit.item.premises_nl) == len(hit.item.premises_fol):
                            for nl, fol in zip(hit.item.premises_nl, hit.item.premises_fol):
                                hit_dict[nl] = fol
                                
                # PARTIAL MATCHING (MIXED NL & FOL):
                # If only some premises exist in the database, we translate the known ones to FOL
                # and keep the unknown ones as NL. The LLM will only have to translate the unknown ones.
                mixed_premises = []
                for p in premises_nl:
                    if p in hit_dict:
                        mixed_premises.append(f"[PRE-TRANSLATED FOL] {hit_dict[p]}")
                    else:
                        mixed_premises.append(f"[NEEDS TRANSLATION] {p}")
                
                matched_fol = mixed_premises

            if matched_fol:
                premises_fol = list(matched_fol)
            else:
                premises_fol = []

        raw_intent = orchestration_data.get("intent", "open_analysis")
        intent = str(raw_intent).strip().lower()
        valid_intents = {"verify_true", "verify_false", "choose_true", "choose_false", "choose_strongest_conclusion", "choose_fewest_premises", "path_finding", "open_analysis"}
        if intent not in valid_intents:
            intent = "open_analysis"
        # Keep the intent contract compatible with the answer shape.  An LLM
        # orchestration response may say verify_true for a multiple-choice
        # question, but the dispatcher requires a choose_* operation.
        if query_type == "multiple_choice" and intent in {"verify_true", "open_analysis"}:
            intent = "choose_true"
        elif query_type == "multiple_choice" and intent == "verify_false":
            intent = "choose_false"
        elif query_type == "yes_no_uncertain" and intent.startswith("choose_"):
            intent = "verify_true"
        orchestration_data["intent"] = intent

        deterministic_result = (
            try_deterministic_logic(
                question=question,
                premises_nl=premises_nl,
                premises_fol=premises_fol,
                options=options,
                query_type=query_type,
                intent=intent,
                learned_aliases=self.predicate_schema.aliases,
            )
            if not requires_full_symbolic
            else None
        )
        if deterministic_result is not None:
            deterministic_result.metadata["route_info"] = route_info
            return deterministic_result

        llm_result = None
        effective_fol: Sequence[str] = premises_fol
        if requires_full_symbolic:
            llm_result = self._answer_with_symbolic_ast(
                question=question,
                premises_nl=premises_nl,
                premises_fol=premises_fol,
                options=options,
                query_type=query_type,
                intent=intent,
                deadline=request_deadline,
            )
        if llm_result is None and not requires_full_symbolic:
            llm_result, effective_fol = self._answer_with_symbolic_logic(
                question,
                premises_nl,
                premises_fol,
                options,
                hits,
                orchestration_data,
            )

        if llm_result is None and not requires_full_symbolic:
            llm_result = self._answer_with_llm(question, premises_nl, effective_fol or premises_fol, hits)
            
        if llm_result is not None:
            if llm_result.metadata.get("executor") == "z3" and not llm_result.metadata.get("execution_errors"):
                # Feedback loop if successful
                code = llm_result.metadata.get("executed_code", "")
                if code:
                    extract_and_write_back_logic(code, self.logic_knowledge_index)
            llm_result.metadata["route_info"] = route_info
            if requires_full_symbolic:
                llm_result.metadata["horn_bypass_reasons"] = list(horn_capability.reasons)
            return llm_result

        if not requires_full_symbolic and hits and hits[0].score >= self.low_match_threshold:
            return self._from_example(hits[0].item, confidence=min(0.58, hits[0].score), source="retrieval-fallback", hit=hits[0])

        fallback_metadata = {"route_info": route_info}
        if requires_full_symbolic:
            fallback_metadata["horn_bypass_reasons"] = list(horn_capability.reasons)
            fallback_metadata["required_executor"] = "z3_symbolic"
        return PipelineResult(
            answer="Uncertain",
            explanation="The premises do not provide enough machine-verifiable support for a definite answer.",
            cot=["Classified the query as Type 1.", "Attempted exact lookup, retrieval, and optional local LLM."],
            premises=premises_nl,
            fol="\n".join(premises_fol) if premises_fol else None,
            confidence=0.2,
            query_type="type1",
            source="fallback",
            metadata=fallback_metadata,
        )

    def _translate_nl_to_fol_chunked(
        self, 
        premises_nl: List[str], 
        question: str = "",
        global_glossary: dict = None,
        pre_translated_context: dict = None
    ) -> List[str]:
        """Translate NL premises to FOL using chunked sentence-by-sentence approach."""
        if not self.llm.enabled:
            return []
        
        CHUNK_SIZE = 3
        
        system_prompt = (
            "You are a strict First-Order Logic (FOL) translator. Translate each premise into exactly one FOL formula.\n"
            f"Target Question/Options for context:\n{question}\n\n"
            "STRICT SYNTAX RULES:\n"
            "1. ONLY use standard logical connectives: ∀, ∃, ¬, ∧, ∨, →, ↔. NEVER use English words like 'if', 'then', 'implies', 'and'.\n"
            "2. EVERY predicate MUST have an argument: P(x) or P(Entity), NOT bare P.\n"
            "2a. Preserve predicate arity and variable binding. Relations may use multiple arguments, "
            "e.g. access(person, facility); do not flatten them into unrelated unary predicates.\n"
            "3. Quantify rules and genuine class statements. Named facts MUST be grounded, "
            "e.g. has_fever(Mira), and MUST NOT be rewritten as a universal statement.\n"
            "4. 'If A then B' must be translated as ∀x(A(x) → B(x)).\n"
            "5. Do NOT invent numbered predicates like l1, l2, l3.\n"
            "6. For math/comparisons, use >, <, >=, <=, =, !=.\n"
            "7. For numerical functions, explicitly define arguments, e.g. score(x) >= 10.\n"
            "8. Constants must be valid identifiers: use MedKit_7, never MedKit-7 or quoted strings.\n"
            "9. Preserve premise boundaries: output exactly one formula for every input premise ID.\n\n"
        )
        
        if global_glossary:
            system_prompt += "GLOBAL GLOSSARY (PRIORITIZE THESE):\n"
            for k, v in global_glossary.items():
                system_prompt += f"- Use predicate '{k}' if the premise relates to '{v}'\n"
            system_prompt += "Reuse glossary predicates for concepts they cover. You may introduce a relation required to preserve entity binding, quantifier scope, or a genuinely new option concept.\n"
        else:
            system_prompt += "3. Use descriptive snake_case(x) (e.g., has_camera(x)).\n"
            
        if pre_translated_context:
            system_prompt += "\nEXISTING TRANSLATIONS (REUSE THESE PREDICATES FOR CONSISTENCY):\n"
            # Limit to 15 to avoid context bloat if there are many
            for idx, (en, fol) in enumerate(pre_translated_context.items()):
                if idx >= 15: break
                system_prompt += f"- \"{en}\" -> {fol}\n"

            
        system_prompt += (
            '\nReturn a JSON object: {"premises_fol": ["formula1", "formula2", ...]}\n'
            "Output ONLY valid JSON. No explanations."
        )
        
        import concurrent.futures

        all_fol = [""] * len(premises_nl)
        
        def translate_chunk(i: int) -> tuple[int, list[str]]:
            chunk = premises_nl[i:i+CHUNK_SIZE]
            chunk_schema = {
                "type": "object",
                "properties": {
                    "premises_fol": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": len(chunk),
                        "maxItems": len(chunk),
                    }
                },
                "required": ["premises_fol"],
                "additionalProperties": False,
            }
            lines = []
            for j, p in enumerate(chunk):
                lines.append(f"P{i + j + 1}: {p}")
            current_chunk_str = "\n".join(lines)
            user_prompt = f"<current_premises>\n{current_chunk_str}\n</current_premises>"
            
            try:
                raw = self.llm.chat_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.0,
                    max_tokens=4096,
                    json_schema=chunk_schema,
                )
                if raw and "premises_fol" in raw:
                    fol_list = list(raw["premises_fol"])
                    if len(fol_list) != len(chunk) or any(not str(f).strip() for f in fol_list):
                        return (
                            i,
                            [f"/* ERROR: invalid one-to-one translation for {p} */" for p in chunk],
                        )
                    cleaned_list = []
                    for f in fol_list:
                        open_c = f.count('(')
                        close_c = f.count(')')
                        if open_c > close_c:
                            f = f + ')' * (open_c - close_c)
                        cleaned_list.append(f)
                    return (i, cleaned_list)
                else:
                    return (i, [f"/* UNTRANSLATED: {p} */" for p in chunk])
            except Exception as exc:
                print(f"[EXACT] Chunk FOL translation error (chunk {i}): {exc}", flush=True)
                return (i, [f"/* ERROR: {p} */" for p in chunk])

        chunk_indices = list(range(0, len(premises_nl), CHUNK_SIZE))
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(chunk_indices) if chunk_indices else 1)) as executor:
            futures = [executor.submit(translate_chunk, i) for i in chunk_indices]
            for future in concurrent.futures.as_completed(futures):
                idx, results = future.result()
                for j, res in enumerate(results):
                    if idx + j < len(all_fol):
                        all_fol[idx + j] = res

        return all_fol

    def _orchestrate_query(self, question: str, query_type: str = "open_ended") -> dict:
        fallback = {
            "exact_entities": [],
            "semantic_anchors": [question],
            "complexity_score": 3,
            "is_solvable": True,
            "intent": "open_analysis",
            "condition": "",
            "target": "",
            "query_type": query_type
        }
        if not self.llm.enabled:
            return fallback
            
        system_prompt = f"""You are an orchestration AI. Read the problem and return ONLY a valid JSON object with the following schema:
- `exact_entities`: List[str] (Key names, named entities, named variables)
- `semantic_anchors`: List[str] (1-2 sentences of HyDE contextual assumptions)
- `complexity_score`: int (1-5, where 1 is a simple lookup, 5 is a complex multi-step logical deduction)
- `is_solvable`: bool (False if it's completely nonsensical or missing required premises)
- `intent`: str. EXACTLY ONE of: 'verify_true', 'verify_false', 'choose_true', 'choose_false', 'choose_strongest_conclusion', 'choose_fewest_premises', 'path_finding', 'open_analysis'.
- `condition`: str. The condition, fact, or rule being assumed before evaluating the main statement, or "" if not explicitly stated.
- `target`: str. Main statement or conclusion being evaluated, or "". It is not the general question, e.g: "Which is the correct conclusion?", return "".

# RULES
- Do NOT change the original query.
- Remove fillers like "does it follow that", "according to the premises".
- The user query type is: {query_type}. If 'yes_no_uncertain', intent is usually verify_true. If 'multiple_choice', pick the choose_* intent. If 'open_ended', pick path_finding or open_analysis.
- CRITICAL: If the question asks "Who", "What", "Which", "How many", it MUST be classified as `open_analysis` and NOT `verify_true` or `verify_false`. It requires finding an exact value, not Yes/No.

----------------
Examples:

Input: Does it follow that if all Python projects are well-structured, then all Python projects are optimized, according to the premises?
Output:
{{
  "exact_entities": ["Python projects"],
  "semantic_anchors": ["Python projects are well-structured", "Python projects are optimized"],
  "complexity_score": 2,
  "is_solvable": true,
  "intent": "verify_true",
  "condition": "all Python projects are well-structured",
  "target": "all Python projects are optimized"
}}

Input: Based on the premises, which statement is most strongly supported?
Output:
{{
  "exact_entities": [],
  "semantic_anchors": ["Find the strongest conclusion"],
  "complexity_score": 3,
  "is_solvable": true,
  "intent": "choose_strongest_conclusion",
  "condition": "",
  "target": ""
}}
"""
        try:
            raw = self.llm.chat_json(
                system_prompt=system_prompt, 
                user_prompt=question,
                temperature=0.0,
                max_tokens=4096
            )
            if raw:
                raw["query_type"] = query_type
                if isinstance(raw, list) and len(raw) > 0:
                    raw = raw[0]
                if isinstance(raw, dict):
                    print(f"\n🚀 [QWEN3-8B ORCHESTRATION] {json.dumps(raw, ensure_ascii=False)}\n", flush=True)
                    return raw
        except Exception as e:
            print(f"[EXACT] Query orchestration failed: {e}", flush=True)
            
        return fallback

    def _fast_path_execute(
        self, question: str, premises_nl: Sequence[str], premises_fol: Sequence[str], payload: dict
    ) -> Optional[PipelineResult]:
        formula_item = self.logic_knowledge_index.fast_path_search(question)
        if not formula_item:
            return None
        
        # In logic, formula_item is a Z3 rule string like "solver.add(Implies(A, B))"
        # We can construct a standalone Z3 script that tries to verify it
        # But realistically, without LLM mapping the NL question to Z3 variables, 
        # evaluating raw generic Z3 formulas on dynamic text is hard.
        # We will attempt to run it if the payload provides variables, otherwise fallback.
        try:
            # Minimal execution wrapper to see if it's purely self-contained or matches payload keys
            code = f"""
import z3
solver = z3.Solver()
# Retrieved verified rule:
{formula_item}
# Try to check sat
if solver.check() == z3.sat:
    RESULT = {{"answer": "Yes", "explanation": "Fast Path logic rule matched."}}
else:
    RESULT = {{"answer": "No Solution", "explanation": "The provided constraints contain contradictions."}}
"""
            executed = self.z3_executor.run(code)
            if executed.ok and executed.answer:
                return PipelineResult(
                    answer=executed.answer,
                    explanation=f"Fast Path evaluated Logic Rule: {formula_item}",
                    cot=[f"Used rule {formula_item} to infer the answer."],
                    confidence=0.95,
                    query_type="type1",
                    source="fast-path-z3",
                    premises=[str(formula_item)]
                )
        except Exception as e:
            print(f"[EXACT] Fast path Z3 execution failed: {e}", flush=True)
            pass
            
        return None

    def _from_example(
        self,
        example: LogicExample,
        *,
        confidence: float,
        source: str,
        hit: Optional[SearchHit[LogicExample]] = None,
    ) -> PipelineResult:
        used_premises = _select_premises(example.premises_nl, example.premise_indices)
        cot = []
        if example.premise_indices:
            cot.append("Use premise indices: " + ", ".join(str(i) for i in example.premise_indices))
        cot.append(example.explanation)
        metadata = {}
        if hit is not None:
            metadata["retrieval_score"] = round(hit.score, 4)
        return PipelineResult(
            answer=example.answer,
            explanation=example.explanation,
            cot=cot,
            premises=used_premises or list(example.premises_nl),
            premises_used=[
                idx - 1
                for idx in example.premise_indices
                if isinstance(idx, int) and 1 <= idx <= len(example.premises_nl)
            ],
            fol="\n".join(example.premises_fol) if example.premises_fol else None,
            confidence=confidence,
            query_type="type1",
            source=source,
            matched_id=example.question_id,
            metadata=metadata,
        )

    def _answer_with_symbolic_ast(
        self,
        *,
        question: str,
        premises_nl: Sequence[str],
        premises_fol: Sequence[str],
        options: Sequence[str],
        query_type: str,
        intent: str,
        deadline: Optional[float] = None,
    ) -> Optional[PipelineResult]:
        """Translate the complete theory to a validated AST, then invoke Z3.

        The production wire representation is a compact prefix AST. Constrained
        decoding guarantees JSON shape, while the deterministic compiler checks
        full token consumption, variable scope, symbol identity, arity and
        polarity. The model never writes solver code or solver syntax.
        """

        ast_started = time.monotonic()
        from exact_pipeline.engines.symbolic_solver import run_symbolic_solver

        clean_fol = [
            str(item).replace("[PRE-TRANSLATED FOL]", "").strip()
            for item in premises_fol
            if str(item).strip()
            and "[NEEDS TRANSLATION]" not in str(item)
            and not str(item).lstrip().startswith("/*")
        ]
        use_fol = bool(clean_fol) and len(clean_fol) == len(premises_fol)
        source_kind = "FOL" if use_fol else "natural language"
        source_premises = clean_fol if use_fol else list(premises_nl)
        option_texts = _extract_symbolic_options(question, options) if query_type == "multiple_choice" else []
        if query_type == "multiple_choice" and not option_texts:
            return None

        # Trusted FOL is executable input, not prose to be reinterpreted.  A
        # consistency preflight catches malformed theories and makes the
        # answer policy for an inconsistent yes/no theory independent of an
        # LLM translation of the question.  Consistent FOL still proceeds to
        # semantic parsing for the natural-language target/options.
        if use_fol:
            preflight = run_symbolic_solver(
                {"query_type": "yes_no_uncertain", "intent": "verify_true"},
                {
                    "predicates": [],
                    "functions": [],
                    "premises_fol": source_premises,
                    "condition_fol": "",
                    "target_fol": "",
                    "options_fol": [],
                },
            )
            preflight_status = preflight.get("verdict")
            if preflight_status == "Error":
                return PipelineResult(
                    answer="Uncertain",
                    explanation=(
                        "The supplied FOL theory is not solver-valid: "
                        + str(preflight.get("explanation", "unknown parser/type error"))
                    ),
                    premises=list(premises_nl) or list(source_premises),
                    fol="\n".join(source_premises),
                    confidence=0.0,
                    query_type="type1",
                    source="symbolic-fol-validation-failed",
                    metadata={
                        "executor": "z3_symbolic_ast",
                        "logical_status": "Error",
                        "symbolic_ir": "trusted-fol-preflight",
                    },
                )
            if preflight_status == "Inconsistent" and query_type == "yes_no_uncertain":
                used_indices = sorted({
                    int(item[1:]) - 1
                    for item in preflight.get("premises_used", [])
                    if isinstance(item, str) and re.fullmatch(r"P\d+", item)
                    and 1 <= int(item[1:]) <= len(source_premises)
                })
                return PipelineResult(
                    answer="Uncertain",
                    explanation=(
                        "The supplied premise theory is inconsistent (UNSAT); under the pipeline's "
                        "three-way policy neither Yes nor No is returned as the unique conclusion."
                    ),
                    premises=list(premises_nl) or list(source_premises),
                    premises_used=used_indices,
                    fol="\n".join(source_premises),
                    confidence=0.99,
                    query_type="type1",
                    source="symbolic-fol-preflight",
                    metadata={
                        "executor": "z3_symbolic_ast",
                        "logical_status": "Inconsistent",
                        "normalization_policy": "inconsistent_to_uncertain",
                        "symbolic_ir": "trusted-fol-preflight",
                    },
                )

        if not self.llm.enabled:
            return None

        cache_key = json.dumps(
            {
                "source_kind": source_kind,
                "premises": source_premises,
                "question": question,
                "options": option_texts,
                "query_type": query_type,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        cache = getattr(self, "_symbolic_ast_cache", None)
        cache_lock = getattr(self, "_symbolic_ast_cache_lock", None)
        cached_compiled = None
        if cache is not None and cache_lock is not None:
            with cache_lock:
                cached_compiled = cache.get(cache_key)
                if cached_compiled is not None:
                    cache.move_to_end(cache_key)

        prompt = (
            "Translate the complete logic problem into the prefix typed-IR JSON contract. Return JSON only.\n"
            "This is semantic parsing only: preserve literal meaning; do not solve, strengthen, weaken, "
            "complete, or repair the premises. Never use an inverse or converse that the text did not state.\n"
            "A formula is one prefix token array. Every atom is ONE string token containing all terms.\n"
            "- Unary atom token: 'p0(x)' or ground atom 'p0(e0)'. Binary atom: 'p1(x,y)'. "
            "Never output separate tokens 'atom','p0','x'.\n"
            "- ['not',FORMULA].\n"
            "- ['and',LEFT,RIGHT], ['or',LEFT,RIGHT], ['implies',LEFT,RIGHT], ['iff',LEFT,RIGHT].\n"
            "- Quantifier is one token: 'forall:x' or 'exists:y', immediately followed by its BODY.\n"
            "- ['none'] is only the absent multiple-choice target.\n"
            "- ['theory_inconsistent'] is only a complete contradiction meta-option.\n"
            "Example: forall x, p0(x)->exists y(p1(x,y) and p2(y)) is exactly "
            "['forall:x','implies','p0(x)','exists:y','and','p1(x,y)','p2(y)'].\n"
            "Example: forall x, p0(x) iff p1(x) is exactly "
            "['forall:x','iff','p0(x)','p1(x)']; do not expand iff into two implications.\n"
            "First build entities: [{id:'e0', mentions:['exact source name']}]. Use only opaque "
            "IDs e0, e1, ... as ground atom terms. Every repeated mention of one real entity must "
            "reuse one ID even when wording adds a type noun (for example 'Med-V4' and "
            "'Model Med-V4'). Declare only named individuals or literal constants as entities. "
            "Generic classes such as 'vehicle', 'secure microgrid', or 'backup generator' are "
            "predicates, never entities; use quantified variables for arbitrary members/witnesses. "
            "Never use predicate names as entity constants.\n"
            "Build a global predicate table too: [{id:'p0', mentions:['passes safety test'], "
            "arity:1}]. Reuse the same p-ID for paraphrases of the same relation across premises, "
            "target and options. Use only declared p-IDs in atoms. Keep argument order stable for "
            "multi-argument relations.\n"
            "Never discard an explicit number, measurement, category, project, person, or other value. "
            "Declare every such literal/named value as an entity eN and preserve it as a relation argument. "
            "For example 'model x has latency 45 milliseconds' is a binary relation p0(x,e45), where "
            "e45 mentions '45 milliseconds'; it is NOT a unary p0(x). Rules must preserve the value slots, "
            "for example p_accuracy(x,e84) -> p_latency(x,e45).\n"
            "Quantify only variables that occur in that body—normally one or two. Never generate "
            "entity_id placeholders or unused variables; named entities are constants in atom terms.\n"
            "Negation MUST use a 'not' token before the positive atom. Never invent "
            "does_not_*, cannot_*, lacks_* or fails_* predicates.\n"
            "Negation scope is local to the clause containing 'not/cannot/no'. In particular, "
            "'any X that has A cannot B' means ['forall:x','implies','A(x)','not','B(x)']; "
            "never negate A. 'No A is B' means A(x) implies not B(x).\n"
            "Use one stable constant ID for every named entity across every premise, target and option.\n"
            "Preserve relations and variable binding with multiple terms; never collapse person/facility or owner/item.\n"
            "Scope conditionals precisely: 'P only if Q' and 'No P unless Q' mean P→Q; "
            "ordinary 'P unless Q' means ¬Q→P.\n"
            "Translate iff as iff, either/or as or, and existence/cardinality as exists. A phrase such as "
            "'has no R' is not(exists y R(...,y)); it is not a new positive predicate.\n"
            "For a multiple-choice meta-option asserting that the premise theory is contradictory/UNSAT, "
            "use ['theory_inconsistent']. Otherwise never use that operator.\n"
            "Output shape: {'translation':{'entities':[],'predicates':[],"
            "'premises':[{'source_index':0,'prefix':[]}],"
            "'target_prefix':[],'options_prefix':[]}}.\n"
            "Emit exactly one premise per source_index. For yes/no, target_prefix is a real formula and "
            "options_prefix is empty. For multiple choice, target_prefix is ['none'] and options_prefix "
            "contains exactly one formula per option in original order.\n\n"
            "For a yes/no question asking whether named entity e0 has property p0, target_prefix must be "
            "['p0(e0)']; never copy a premise rule into the target.\n\n"
            "For an open-ended Who/Which/What-value query, use the reserved term answer exactly in the "
            "requested argument position and nowhere in the premises. Examples: 'Which developer is assigned?' "
            "uses target_prefix ['p0(answer)']; 'What is model e0 latency?' uses "
            "target_prefix ['p1(e0,answer)']. Declare all possible named/numeric answer constants in "
            "entities with faithful source mentions. Do not answer the question yourself.\n\n"
            f"Source format: {source_kind}\n"
            f"Premises (zero-based source indices):\n"
            + "\n".join(f"P{index}: {premise}" for index, premise in enumerate(source_premises))
            + f"\n\nQuestion: {question}\nQuery type: {query_type}\nIntent: {intent}\n"
            + ("Options:\n" + "\n".join(
                f"O{index}: {option}" for index, option in enumerate(option_texts)
            ) if option_texts else "Options: none")
        )

        errors: List[str] = []
        rejected_output: Optional[dict] = None
        ir_kind = "prefix-typed-ir-v1"
        max_attempts = 1 if cached_compiled is not None else min(self.max_retries + 1, 2)
        for attempt in range(max_attempts):
            remaining = (deadline - time.monotonic()) if deadline is not None else 24.0
            if remaining <= 2.0:
                errors.append("symbolic request deadline exhausted")
                break
            user_prompt = prompt
            if errors:
                user_prompt += (
                    "\n\nThe previous semantic AST was rejected. Rebuild the complete translation; "
                    "fix only the stated contract violations and preserve source meaning:\n- "
                    + "\n- ".join(errors[-3:])
                )
                if rejected_output is not None:
                    rejected_preview = json.dumps(
                        rejected_output, ensure_ascii=False, separators=(",", ":")
                    )
                    user_prompt += "\nRejected output:\n" + rejected_preview[:6000]
            try:
                if cached_compiled is not None:
                    compiled = cached_compiled
                else:
                    guided = attempt == 0 and query_type != "open_ended"
                    if query_type == "open_ended":
                        max_output_tokens = 900
                        timeout_cap = 38.0 if attempt == 0 else 14.0
                    else:
                        max_output_tokens = 2048 if guided else 1536
                        timeout_cap = 28.0 if guided else 20.0
                    raw = self.llm.chat_json(
                        system_prompt=(
                            "You are a strict first-order-logic semantic parser. Return JSON only. "
                            "Never answer the problem directly."
                        ),
                        user_prompt=user_prompt,
                        temperature=0.0,
                        max_tokens=max_output_tokens,
                        # Prefix IR keeps the guided grammar shallow. If the
                        # serving backend still times out compiling/running
                        # guided decoding, the bounded retry uses the same
                        # validated contract without server-side guidance.
                        json_schema=logic_prefix_ir_json_schema() if guided else None,
                        request_timeout_s=min(timeout_cap, max(1.0, remaining - 2.0)),
                    )
                    raw = coerce_translation_envelope(raw)
                    rejected_output = dict(raw) if isinstance(raw, dict) else None
                    raw_translation = raw.get("translation", {}) if isinstance(raw, dict) else {}
                    raw_premises = raw_translation.get("premises", []) if isinstance(raw_translation, dict) else []
                    first_premise = raw_premises[0] if raw_premises and isinstance(raw_premises[0], dict) else {}
                    # The constrained production protocol is flat AST.  The
                    # other two typed/validated readers remain accepted for
                    # rolling upgrades and already-populated external caches.
                    if "prefix" in first_premise:
                        compiler = compile_prefix_ir_translation
                        ir_kind = "prefix-typed-ir-v1"
                    elif "formula" in first_premise:
                        compiler = compile_flat_ast_translation
                        ir_kind = "flat-typed-ast-v1"
                    elif "ast" in first_premise:
                        compiler = compile_ast_translation
                        ir_kind = "nested-typed-ast-v1"
                    elif "fol" in first_premise:
                        compiler = compile_compact_fol_translation
                        ir_kind = "validated-compact-fol-v1"
                    else:
                        raise AstValidationError("premises do not contain a recognized typed formula representation")
                    compiled = compiler(
                        raw,
                        premise_count=len(source_premises),
                        query_type=query_type,
                        option_count=len(option_texts),
                    )
                    if source_kind == "natural language":
                        semantic_errors = _validate_compiled_semantic_alignment(
                            premises_nl=source_premises,
                            option_texts=option_texts,
                            question=question,
                            query_type=query_type,
                            compiled=compiled,
                        )
                        if semantic_errors:
                            raise AstValidationError(
                                "semantic alignment: " + "; ".join(semantic_errors[:8])
                            )
                    if cache is not None and cache_lock is not None:
                        with cache_lock:
                            cache[cache_key] = compiled
                            cache.move_to_end(cache_key)
                            capacity = getattr(self, "_symbolic_ast_cache_capacity", 256)
                            while len(cache) > capacity:
                                cache.popitem(last=False)
            except LLMError as exc:
                errors.append(f"AST translation request: {exc}")
                # Retry once without guided decoding while the request-level
                # deadline still leaves a meaningful inference budget.
                remaining_after_error = (
                    deadline - time.monotonic() if deadline is not None else 0.0
                )
                if attempt + 1 >= max_attempts or remaining_after_error <= 5.0:
                    break
                continue
            except (AstValidationError, TypeError, ValueError) as exc:
                errors.append(f"AST validation: {exc}")
                continue

            translation = {
                "predicates": list(compiled.predicates),
                "functions": [],
                "premises_fol": list(compiled.premises_fol),
                "condition_fol": "",
                "target_fol": compiled.target_fol,
                "options_fol": list(compiled.options_fol),
                "constant_labels": {
                    entity_id: mentions[0]
                    for entity_id, mentions in compiled.entity_mentions
                    if mentions
                },
            }
            solver_result = run_symbolic_solver(
                {
                    "query_type": query_type,
                    "intent": intent,
                    # Translation retries and SMT solving share one request
                    # budget.  The solver must not start a fresh seven-second
                    # allowance after the outer deadline is nearly exhausted.
                    "deadline_monotonic": deadline,
                },
                translation,
            )
            verdict = solver_result.get("verdict", "Uncertain")
            if verdict == "Error":
                errors.append(f"solver: {solver_result.get('explanation', 'unknown error')}")
                continue
            used_indices = sorted({
                int(item[1:]) - 1
                for item in solver_result.get("premises_used", [])
                if isinstance(item, str) and re.fullmatch(r"P\d+", item)
                and 1 <= int(item[1:]) <= len(source_premises)
            })
            normalization_policy = "direct"
            answer_unit = ""
            if query_type == "open_ended":
                extracted_answer = solver_result.get("answer")
                if verdict == "Answer" and extracted_answer:
                    answer = str(extracted_answer)
                    answer_unit = str(solver_result.get("unit", ""))
                    normalization_policy = "symbolic_answer_projection"
                else:
                    answer = "Uncertain"
                    normalization_policy = "no_entailed_answer_candidate"
            elif query_type == "multiple_choice":
                best_option = solver_result.get("best_option")
                if best_option:
                    answer = str(best_option)
                else:
                    answer = "Uncertain"
                    normalization_policy = "no_supported_option_to_uncertain"
            elif verdict == "True":
                answer = "Yes"
            elif verdict == "False":
                answer = "No"
            else:
                answer = "Uncertain"
                if verdict == "Inconsistent":
                    normalization_policy = "inconsistent_to_uncertain"
            rendered_explanation, rendered_reasoning = _render_symbolic_explanation(
                verdict=verdict,
                answer=answer,
                question=question,
                query_type=query_type,
                option_texts=option_texts,
                premises_nl=list(premises_nl) or list(source_premises),
                used_indices=used_indices,
                unit=answer_unit,
            )
            return PipelineResult(
                answer=answer,
                explanation=rendered_explanation,
                unit=answer_unit,
                premises=list(premises_nl) or list(source_premises),
                premises_used=used_indices,
                reasoning=rendered_reasoning,
                fol="\n".join(compiled.premises_fol),
                confidence=0.94 if verdict in {"True", "False", "Answer"} else 0.78,
                query_type="type1",
                source="symbolic-ast-z3",
                metadata={
                    "executor": "z3_symbolic_ast",
                    "translation_attempts": attempt + 1,
                    "translation_errors": errors,
                    "constants": list(compiled.constants),
                    "symbolic_elapsed_s": round(time.monotonic() - ast_started, 3),
                    "symbolic_cache_hit": cached_compiled is not None,
                    "symbolic_ir": ir_kind,
                    "logical_status": verdict,
                    "normalization_policy": normalization_policy,
                },
            )
        return PipelineResult(
            answer="Uncertain",
            explanation=(
                "The symbolic semantic parse could not be validated within the request budget; "
                "no unverified translation was sent to Z3."
                + (
                    " Errors: " + " | ".join(error[:220] for error in errors[-2:])
                    if errors else ""
                )
            ),
            premises=list(premises_nl) or list(source_premises),
            confidence=0.0,
            query_type="type1",
            source="symbolic-ast-validation-failed",
            metadata={
                "executor": "z3_symbolic_ast",
                "translation_errors": errors,
                "symbolic_elapsed_s": round(time.monotonic() - ast_started, 3),
                "symbolic_ir": ir_kind,
            },
        )

    def _answer_with_symbolic_logic(
        self,
        question: str,
        premises_nl: Sequence[str],
        premises_fol: Sequence[str],
        options: Sequence[str],
        hits: Sequence[SearchHit[LogicExample]],
        orchestration_data: dict,
    ) -> tuple[Optional[PipelineResult], Sequence[str]]:
        if not self.llm.enabled:
            return None, premises_fol

        semantics = {
            "query_type": orchestration_data.get("query_type", "open_ended"),
            "intent": orchestration_data.get("intent", "open_analysis"),
            "condition": orchestration_data.get("condition", ""),
            "target": orchestration_data.get("target", "")
        }

        # Extract Options FOL from question text (simple inline regex)
        options_fol_extracted = (
            _extract_symbolic_options(question, options)
            if semantics.get("query_type") == "multiple_choice"
            else []
        )
        opt_regex = r'(?:^|\n)\s*[A-Z][.)\s]+(.+?)(?=\n\s*[A-Z][.)\s]|\Z)'
                
        # Clean question for translator (strip options dynamically to avoid hallucination)
        clean_question = re.sub(opt_regex, '', question, flags=re.DOTALL).strip()
                
        # 1. DYNAMIC PREDICATE STANDARDIZER (GLOBAL GLOSSARY)
        global_glossary = {}
        if self.llm.enabled:
            print("\n🔍 [GLOSSARY] Extracting Global Glossary from Context...", flush=True)
            schema = {
                "type": "object",
                "properties": {
                    "glossary": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Map of standard snake_case predicates to their English meaning. e.g. {'gps(x)': 'represents anything related to gps or having gps'}"
                    }
                },
                "required": ["glossary"],
                "additionalProperties": False
            }
            all_premises = "\n".join(premises_nl)
            prompt = (
                f"Context premises:\n{all_premises}\n"
                f"Question:\n{clean_question}\n\n"
                "Extract a JSON glossary mapping standardized snake_case predicates (e.g. 'gps(x)', 'mammal(x)') "
                "to their core English keyword meaning. Group similar concepts under the same predicate to avoid duplicates.\n"
                "CRITICAL: Each predicate MUST be unique and concise. The key MUST be in the format `predicate(x)`.\n"
                "CRITICAL: Extract properties and relations only. Entity names, constants, pronouns, "
                "and temporal/deictic modifiers are arguments or context—not predicates.\n"
                "CRITICAL: Do NOT extract full sentences or full premises as predicates (e.g., do NOT output 'a_is_true(x)'). "
                "If the premise is 'A is true', the predicate is 'A(x)'. If 'Fluffy is a mammal', the predicate is 'mammal(x)'. "
                "Do NOT use variations like 'has_gps(x)' and 'gps(x)' simultaneously; pick one standard name."
            )
            try:
                res = self.llm.chat_json(
                    system_prompt="You are a strict logic glossary builder. Output ONLY valid JSON.",
                    user_prompt=prompt,
                    temperature=0.0,
                    max_tokens=4096,
                    json_schema=schema
                )
                if res and "glossary" in res:
                    global_glossary = res["glossary"]
                    print(f"   => {global_glossary}")
            except Exception as e:
                print(f"   => Failed to extract glossary: {e}")

        # 2. CHUNKED TRANSLATION
        if premises_fol is None:
            premises_fol = []
        needs_translation = not premises_fol or any("[NEEDS TRANSLATION]" in f for f in premises_fol)
        effective_fol = list(premises_fol)
        
        if needs_translation:
            if premises_fol and any("[NEEDS TRANSLATION]" in f for f in premises_fol):
                print(f"\n📝 [PARTIAL TRANSLATION] Found cached FOL. Translating missing premises...", flush=True)
                missing_indices = []
                missing_nl = []
                pre_translated_context = {}
                for i, p in enumerate(premises_fol):
                    if "[NEEDS TRANSLATION]" in p:
                        missing_indices.append(i)
                        missing_nl.append(p.replace("[NEEDS TRANSLATION]", "").strip())
                    elif "[PRE-TRANSLATED FOL]" in p:
                        fol_str = p.replace("[PRE-TRANSLATED FOL]", "").strip()
                        if i < len(premises_nl):
                            pre_translated_context[premises_nl[i]] = fol_str
                
                translated_missing = self._translate_nl_to_fol_chunked(
                    missing_nl, clean_question, global_glossary, pre_translated_context
                )
                
                for i, idx in enumerate(missing_indices):
                    if i < len(translated_missing):
                        effective_fol[idx] = translated_missing[i]
                    else:
                        effective_fol[idx] = "/* ERROR: Missing translation */"
                        
                for i in range(len(effective_fol)):
                    if "[PRE-TRANSLATED FOL]" in effective_fol[i]:
                        effective_fol[i] = effective_fol[i].replace("[PRE-TRANSLATED FOL]", "").strip()
            else:
                print(f"\n📝 [CHUNKED TRANSLATION] Starting for {len(premises_nl)} premises...", flush=True)
                effective_fol = self._translate_nl_to_fol_chunked(list(premises_nl), clean_question, global_glossary)
            
            print(f"📝 [CHUNKED TRANSLATION] Got {len(effective_fol)} FOL formulas", flush=True)
            for idx, fol in enumerate(effective_fol):
                print(f"   P{idx+1}: {fol}", flush=True)
        else:
            # If purely pre-translated with tags, strip them
            for i in range(len(effective_fol)):
                if "[PRE-TRANSLATED FOL]" in effective_fol[i]:
                    effective_fol[i] = effective_fol[i].replace("[PRE-TRANSLATED FOL]", "").strip()

        # POST-PROCESSING: Sanitize FOL formulas to fix common LLM translation errors
        # This catches tautologies (P(x)→P(x)), double-applications (pred(x)(x)), etc.
        effective_fol = _sanitize_fol_list(effective_fol)

        # Collect unique predicates from the translated FOL
        # 4. PREPARE SOLVER PAYLOAD & TRANSLATE OPTIONS
        print("\n" + "-"*60)
        print("⚙️ STEP 4: Z3 SYMBOLIC SOLVER (with Retry Loop)")
        print("-"*60)
        
        all_preds = set()
        for fol_str in effective_fol:
            found = re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*)\s*\(', fol_str)
            all_preds.update(found)
        # Remove common non-predicates
        all_preds -= {'x', 'y', 'z'}
        
        print(f"🔧 [AUTO-DETECTED PREDICATES] {sorted(all_preds)}", flush=True)
        print(f"🔧 [OPTIONS FOL (ENGLISH)] {options_fol_extracted}", flush=True)

        from exact_pipeline.llm.templates import LOGIC_SYMBOLIC_TEMPLATE
        from exact_pipeline.engines.symbolic_solver import run_symbolic_solver
        
        errors: List[str] = []
        translation = None
        
        for attempt in range(self.max_retries + 1):
            if attempt == 0 or errors:
                action_str = f"Re-translating" if attempt > 0 else "Translating options/target"
                print(f"\n🔄 [ATTEMPT {attempt}] {action_str} with full context...", flush=True)
                sys_prompt = LOGIC_SYMBOLIC_TEMPLATE.render(
                    question=clean_question, 
                    premises=list(premises_nl), 
                    premises_fol=effective_fol,
                    semantics=semantics,
                    options=options_fol_extracted,
                    global_glossary=global_glossary
                )
                
                if attempt == 0:
                    user_prompt = "Translate the question, target, and options into FOL. Preserve the existing premises_fol exactly."
                else:
                    user_prompt = (
                        "Translate the question and extract semantics as JSON.\n"
                        f"Previous solver error:\n{errors[-1]}\nPlease fix your FOL syntax."
                    )
                
                retry_schema = {
                    "type": "object",
                    "properties": {
                        "translation": {
                            "type": "object",
                            "properties": {
                                "predicates": {"type": "array", "items": {"type": "string"}},
                                "functions": {"type": "array", "items": {"type": "string"}},
                                "premises_fol": {"type": "array", "items": {"type": "string"}},
                                "condition_fol": {"type": "string"},
                                "target_fol": {"type": "string"},
                                "options_fol": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["predicates", "premises_fol", "condition_fol", "target_fol", "options_fol"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["translation"],
                    "additionalProperties": False
                }
                
                try:
                    raw = self.llm.chat_json(
                        system_prompt=sys_prompt, 
                        user_prompt=user_prompt, 
                        temperature=0.2, 
                        max_tokens=4096,
                        json_schema=retry_schema
                    )
                    if raw and "translation" in raw:
                        translation = raw["translation"]
                except Exception as exc:
                    print(f"[EXACT] LLM symbolic translation error: {exc}", flush=True)

            if not translation:
                break

            contract_errors = _validate_symbolic_translation_contract(
                translation,
                premise_count=len(effective_fol),
                query_type=str(semantics.get("query_type", "")),
                option_count=len(options_fol_extracted),
            )
            if contract_errors:
                errors.append("; ".join(contract_errors))
                print(f"[EXACT] Invalid symbolic translation (attempt {attempt+1}): {errors[-1]}", flush=True)
                continue

            # Normalize every solver-bound formula, including formulas echoed
            # by the legacy translator. Lexical negative predicate names are
            # converted to classical negation before parsing.
            translation = dict(translation)
            translation["premises_fol"] = _sanitize_fol_list(translation["premises_fol"])
            translation["target_fol"] = _sanitize_fol_formula(translation.get("target_fol", ""))
            translation["options_fol"] = _sanitize_fol_list(translation.get("options_fol", []))

            solver_res = run_symbolic_solver(semantics, translation)
            verdict = solver_res.get("verdict", "Uncertain")
            
            if verdict == "Error":
                errors.append(solver_res.get("explanation", "Unknown parsing error"))
                print(f"[EXACT] Symbolic solver error (attempt {attempt+1}): {errors[-1]}", flush=True)
                if semantics.get("query_type") == "open_ended":
                    break
                continue

            if verdict == "Uncertain":
                print(f"[EXACT] Symbolic solver uncertain (attempt {attempt+1}): {solver_res.get('explanation', 'Uncertain result')}", flush=True)
                if (
                    semantics.get("query_type") == "yes_no_uncertain"
                    and translation.get("target_fol", "").strip()
                ):
                    used_indices = []
                    for p in solver_res.get("premises_used", []):
                        if isinstance(p, str) and p.startswith("P"):
                            try:
                                used_indices.append(int(p[1:]) - 1)
                            except ValueError:
                                pass
                    used_indices = sorted(list(set([i for i in used_indices if 0 <= i < len(premises_nl)])))
                    return PipelineResult(
                        answer="Uncertain",
                        unit="",
                        explanation="The symbolic solver found that neither the target nor its negation is entailed by the premises.",
                        cot=[
                            f"Chunked FOL Translation ({len(effective_fol)} premises)",
                            f"Auto-detected predicates: {sorted(all_preds)}",
                            f"Solver Explanation: {solver_res.get('explanation')}",
                        ],
                        premises=list(premises_nl),
                        premises_used=used_indices,
                        fol="\n".join(translation.get("premises_fol", [])),
                        confidence=0.82,
                        query_type="type1",
                        source="symbolic-solver",
                        metadata={"executor": "z3_symbolic", "semantics": semantics, "global_glossary": global_glossary, "translation": translation},
                    ), effective_fol
                if semantics.get("query_type") == "open_ended":
                    # Symbolic solver cannot solve open_ended, so no point in retrying.
                    # Break out of the retry loop to immediately fall back to the LLM solver.
                    break
                # A valid formal theory may genuinely entail no offered
                # option. Retrying then would bias translation toward an answer.
                return PipelineResult(
                    answer="Uncertain",
                    unit="",
                    explanation=solver_res.get("explanation", "No option is entailed."),
                    cot=[f"Solver Explanation: {solver_res.get('explanation', '')}"],
                    premises=list(premises_nl),
                    premises_used=[],
                    fol="\n".join(translation.get("premises_fol", [])),
                    confidence=0.82,
                    query_type="type1",
                    source="symbolic-solver",
                    metadata={"executor": "z3_symbolic", "semantics": semantics, "global_glossary": global_glossary, "translation": translation},
                ), effective_fol

            cot = [
                f"Chunked FOL Translation ({len(effective_fol)} premises)",
                f"Auto-detected predicates: {sorted(all_preds)}",
                f"Solver Explanation: {solver_res.get('explanation')}",
            ]
            
            # Map solver tracking strings (e.g. "P1", "P2") back to 0-based indices
            used_indices = []
            for p in solver_res.get("premises_used", []):
                if isinstance(p, str) and p.startswith("P"):
                    try:
                        used_indices.append(int(p[1:]) - 1)
                    except ValueError:
                        pass
            
            # Remove any out-of-bounds indices
            used_indices = sorted(list(set([i for i in used_indices if 0 <= i < len(premises_nl)])))

            if semantics.get("query_type") == "yes_no_uncertain":
                if verdict == "True":
                    final_answer = "Yes"
                elif verdict == "False":
                    final_answer = "No"
                else:
                    final_answer = verdict
                
                # If there are explicit multiple choice options, map the answer to the option letter
                if options_fol_extracted:
                    for i, opt_text in enumerate(options_fol_extracted):
                        if final_answer.lower() == opt_text.strip().lower():
                            final_answer = chr(65 + i)
                            break
            else:
                final_answer = str(solver_res.get("best_option", verdict))
            
            if used_indices:
                used_str = "\n".join([f"- {premises_nl[i]}" for i in used_indices])
                explanation = f"The conclusion {final_answer} is proven to be logically and mathematically valid based on the following premises:\n{used_str}"
            else:
                explanation = solver_res.get("explanation", "")

            return PipelineResult(
                answer=final_answer,
                unit="",
                explanation=explanation,
                cot=cot,
                premises=list(premises_nl),
                premises_used=used_indices,
                fol="\n".join(translation.get("premises_fol", [])),
                confidence=0.9,
                query_type="type1",
                source="symbolic-solver",
                metadata={"executor": "z3_symbolic", "semantics": semantics, "global_glossary": global_glossary, "translation": translation}
            ), effective_fol

        return None, effective_fol

    def _answer_with_llm(
        self,
        question: str,
        premises_nl: Sequence[str],
        premises_fol: Sequence[str],
        hits: Sequence[SearchHit[LogicExample]],
    ) -> Optional[PipelineResult]:
        if not self.llm.enabled:
            return None

        examples_text = render_hits(hits, logic_document)
        
        from exact_pipeline.llm.templates import LOGIC_TEMPLATE


        # If FOL is available, exclusively use it to avoid attention distraction.
        # Fallback to NL only if FOL is missing or empty.
        if premises_fol:
            display_premises = "Premises-FOL:\n" + "\n".join(premises_fol)
            premises_list = list(premises_fol)
        else:
            display_premises = "Premises-NL:\n" + "\n".join(f"{i + 1}. {premise}" for i, premise in enumerate(premises_nl))
            premises_list = list(premises_nl)
            
        system_prompt = LOGIC_TEMPLATE.render(question=question, premises=premises_list)
        subgraph_context_lines = get_reasoning_subgraph_context(question, self.logic_knowledge_index, max_cards=4)
        subgraph_text = "\n".join(subgraph_context_lines)

        base_prompt = (
            f"{display_premises}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Retrieved Logic Rules & Reasoning Subgraph:\n"
            f"{subgraph_text}\n\n"
            "Retrieved examples:\n"
            f"{examples_text}"
        )
        
        last_raw: Optional[dict] = None
        errors: List[str] = []
        for attempt in range(self.max_retries + 1):
            user_prompt = base_prompt
            if errors:
                user_prompt += (
                    "\n\nPrevious Z3/code execution error:\n"
                    + errors[-1]
                    + "\nRepair only the formalization/code. Keep the answer grounded in the premises."
                )
            try:
                raw = self.llm.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    thinking=False,
                )
            except LLMError as exc:
                print(f"[EXACT] LLM error: {exc}", flush=True)
                return self._llm_logic_fallback(last_raw, premises_nl, premises_fol, errors) if last_raw else None
            if not raw:
                continue
            last_raw = raw
            z3_code = str(raw.get("python_code") or raw.get("code") or "")
            if not z3_code.strip():
                break

            print(f"\\n[DEBUG] LLM Generated Code:\\n{z3_code}\\n", flush=True)
            executed = self.z3_executor.run(z3_code)
            if executed.ok:
                return self._from_logic_execution(raw, executed, premises_nl, premises_fol, attempt + 1, question=question)
            errors.append(executed.error)
            if executed.error.startswith("z3_unavailable"):
                break

        return self._llm_logic_fallback(last_raw, premises_nl, premises_fol, errors)

    def _from_logic_execution(
        self,
        raw: dict,
        executed: ExecutionResult,
        premises_nl: Sequence[str],
        premises_fol: Sequence[str],
        attempts: int,
        question: str = "",
    ) -> PipelineResult:
        raw_cot = _ensure_list(raw.get("cot", []))
        cot = raw_cot + executed.cot
        if executed.stdout:
            cot.append("Executed generated Z3/Python verifier and parsed its RESULT payload.")
        premises = merge_premises(executed.premises, _ensure_list(raw.get("premises", [])), premises_nl)
        fol = executed.fol or raw.get("fol") or ("\n".join(premises_fol) if premises_fol else None)
        
        # --- ANSWER NORMALIZATION ---
        # If the LLM returned a text answer (e.g. "Yes") but the question has
        # options (A. Yes, B. No, C. Uncertain), map it to the option letter.
        answer = executed.answer or str(raw.get("answer", "Uncertain"))
        if question and answer.strip() and len(answer.strip()) > 1:
            # Extract options from question text
            opt_regex = r'(?:^|\n)\s*([A-Z])[.)\s]+(.+?)(?=\n\s*[A-Z][.)\s]|\Z)'
            opt_matches = re.findall(opt_regex, question, re.DOTALL)
            if opt_matches:
                answer_lower = answer.strip().lower()
                for letter, text in opt_matches:
                    if text.strip().lower() == answer_lower:
                        answer = letter
                        break
        validated_answer = _validate_logic_answer_shape(answer, question)
        answer_rejected = validated_answer is None
        answer = validated_answer or "Uncertain"
        metadata = {
            "llm_attempts": attempts,
            "executor": "z3",
            "executed_code": raw.get("python_code", raw.get("code", "")),
            **executed.metadata,
        }
        if answer_rejected:
            metadata["rejected_answer"] = str(executed.answer or raw.get("answer", ""))
            metadata["output_validation"] = "failed"
        return PipelineResult(
            answer=answer,
            unit="",
            explanation=executed.explanation
            or str(raw.get("explanation", ""))
            or "The generated verifier answered using the supplied logical premises.",
            cot=cot,
            premises=premises,
            premises_used=raw.get("premises_used", []),
            fol=fol,
            confidence=0.25 if answer_rejected else float(raw.get("confidence", 0.68) or 0.68),
            query_type="type1",
            source="self-hosted-llm-z3",
            metadata=metadata,
        )

    def _llm_logic_fallback(
        self,
        raw: Optional[dict],
        premises_nl: Sequence[str],
        premises_fol: Sequence[str],
        errors: Sequence[str],
    ) -> Optional[PipelineResult]:
        if not raw:
            return None
        metadata = {"executor": "z3", "fallback_reason": "execution_failed_or_not_provided"}
        if raw and (raw.get("python_code") or raw.get("code")):
            metadata["failed_code"] = raw.get("python_code") or raw.get("code")
        if errors:
            metadata["execution_errors"] = list(errors[-2:])
        return PipelineResult(
            answer="Uncertain",
            explanation="The generated formalization was not executable, so its unverified answer was rejected.",
            cot=_ensure_list(raw.get("cot", [])),
            premises=merge_premises(_ensure_list(raw.get("premises", [])), premises_nl),
            fol=raw.get("fol") or ("\n".join(premises_fol) if premises_fol else None),
            confidence=min(float(raw.get("confidence", 0.52) or 0.52), 0.62),
            query_type="type1",
            source="self-hosted-llm-fallback",
            metadata=metadata,
        )


def _extract_symbolic_options(question: str, options: Sequence[str]) -> List[str]:
    """Return option texts from either the inline question or payload choices."""

    pattern = r'(?:^|\n)\s*[A-Z][.)\s]+(.+?)(?=\n\s*[A-Z][.)\s]|\Z)'
    inline = [match.strip() for match in re.findall(pattern, question, re.DOTALL) if match.strip()]
    if inline:
        return inline
    extracted: List[str] = []
    for raw_option in options or ():
        clean = re.sub(r"^\s*[A-Z][.)]\s*", "", str(raw_option)).strip()
        if clean and not re.fullmatch(r"[A-Z]", clean):
            extracted.append(clean)
    return extracted


def _ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def _validate_logic_answer_shape(answer: object, question: str) -> Optional[str]:
    """Enforce the answer type implied by the question before API emission."""

    value = str(answer or "").strip()
    if not value:
        return None
    option_matches = re.findall(
        r"(?:^|\n)\s*([A-Z])[\.)\s]+(.+?)(?=\n\s*[A-Z][\.)\s]|\Z)",
        question,
        flags=re.DOTALL,
    )
    if option_matches:
        letters = {letter for letter, _ in option_matches}
        if value.upper() in letters:
            return value.upper()
        matching = [
            letter
            for letter, option_text in option_matches
            if value.lower() == option_text.strip().lower()
        ]
        return matching[0] if len(matching) == 1 else None

    cleaned_question = _clean_question_for_answer_contract(question)
    if re.match(r"^(?:is|are|was|were|does|do|did|can|may|must|should|will)\b", cleaned_question, flags=re.I):
        normalized = value.capitalize()
        return normalized if normalized in {"Yes", "No", "Uncertain"} else None
    if re.match(r"^(?:how many|in how many|what is|at what time)\b", cleaned_question, flags=re.I):
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?|\d{1,2}:\d{2}", value):
            return value
        return None
    if re.match(r"^(?:which|who)\b", cleaned_question, flags=re.I):
        entities = [part.strip() for part in value.split(",")]
        if entities and all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*)*", part) for part in entities):
            return ", ".join(entities)
        return None
    return value if len(value) <= 120 and "\n" not in value else None


def _clean_question_for_answer_contract(question: str) -> str:
    return re.sub(r"\s+", " ", str(question or "").strip()).strip(" .?")


def _select_premises(premises: Sequence[str], indices: Sequence[int]) -> List[str]:
    selected: List[str] = []
    for index in indices:
        if 1 <= index <= len(premises):
            selected.append(f"Premise {index}: {premises[index - 1]}")
    return selected


def _unique_or_ambiguous(mapping: Dict[str, Optional[LogicExample]], key: str, example: LogicExample) -> Optional[LogicExample]:
    if key not in mapping:
        return example
    existing = mapping[key]
    if existing is None:
        return None
    if existing.answer == example.answer and existing.explanation == example.explanation:
        return existing
    return None
