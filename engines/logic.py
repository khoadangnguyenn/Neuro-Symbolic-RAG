"""Type 1 logic-based educational QA pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import math
import re

from exact_pipeline.core.data import logic_document
from exact_pipeline.engines.executors import ExecutionResult, Z3Executor
from exact_pipeline.knowledge.knowledge import get_logic_knowledge_index, merge_premises, render_logic_knowledge, get_reasoning_subgraph_context
from exact_pipeline.llm.llm import LLMError, OpenAICompatibleLLM
from exact_pipeline.core.models import LogicExample, PipelineResult
from exact_pipeline.knowledge.retrieval import SearchHit, VectorDBIndex, render_hits, CrossEncoderReranker
from exact_pipeline.utils.text_utils import first_present, normalize_key
from exact_pipeline.llm.templates import LOGIC_TEMPLATE
from exact_pipeline.orchestration.feedback import extract_and_write_back_logic
from exact_pipeline.knowledge.graph_db import HybridDB


TRANSLATION_SYSTEM_PROMPT = """
### ROLE
You are an expert logic translator and schema extractor. Your task is a two-step process:
1. Extract a unified, consistent, and semantically deduplicated set of predicates from all natural language premises.
2. Convert the premises into First-Order Logic (FOL) formulas using ONLY the extracted predicates.

### RULES
1. Standard FOL Symbols:
- Universal quantifier: ∀
- Existential quantifier: ∃
- Negation: ¬
- Conjunction: ∧
- Disjunction: ∨
- Implication: →
- Biconditional: ↔
- Equality: =

2. Predicate Extraction & Naming: 
- Before translating, analyze all premises to identify all unique properties or relations.
- Format all predicates using `snake_case` with their bound variable (e.g., `well_tested(x)`).
- List all extracted unique predicates in the `"predicates"` array of the JSON output.

3. Strict Semantic Unification (Anti-Mismatch):
- Actively detect and merge phrases that share the SAME semantic meaning but use different wording, synonyms, or syntactic structures (e.g., active vs. passive voice, nouns vs. adjectives).
- Examples of semantic pairs that MUST be unified into a single predicate:
  * "has a good structure" AND "is well-structured" -> `well_structured(x)`
  * "code is clean and readable" AND "has clean code" -> `clean_code(x)`
  * "follows PEP 8" AND "is compliant with PEP 8 standards" -> `follow_pep_8(x)`
- Choose the most concise and representative name for the unified predicate.

4. Strict Vocabulary Adherence: 
- The FOL formulas in the `"premises_fol"` array MUST strictly and exclusively use the exact predicates declared in the `"predicates"` field. 
- Do NOT alter predicate names, change their variable structure, or introduce any unlisted predicates during the translation step.

5. Quantifier Scope & Variable Binding: 
- Every predicate must be associated with a variable in parentheses (e.g., P(x)). Do not use predicates without variables.
- Ensure the scope of all quantifiers (∀, ∃) is clearly defined using parentheses to encompass the entire logical expression.

6. Implicit Universe of Discourse (Domain): 
- Identify the overarching subject matter or base entity type of the context (e.g., "Python code", "patients", "employees"). 
- Do NOT create explicit predicates for this base domain itself (e.g., do NOT create `python_project(x)` or `employee(x)`).
- Universal claims about the domain ("All [Entities] are [Property]") translate directly as: ∀x (property(x))
- Existential claims about the domain ("There exists an [Entity] that is [Property]") translate directly as: ∃x (property(x))

7. Output Format: 
Return ONLY a valid JSON object. Do not provide explanations, conversational fillers, or markdown code blocks.
The JSON must strictly follow this schema:
{
  "predicates": ["predicate_1(x)", "predicate_2(x)"],
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


class LogicPipeline:
    def __init__(
        self,
        examples: Sequence[LogicExample],
        *,
        db_path: str,
        graph_path: str,
        alpha: float,
        llm: Optional[OpenAICompatibleLLM] = None,
        expansion_llm: Optional[OpenAICompatibleLLM] = None,
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
        self.expansion_llm = expansion_llm or self.llm
        self.retrieval_k = retrieval_k
        self.high_match_threshold = high_match_threshold
        self.low_match_threshold = low_match_threshold
        self.max_retries = max(0, max_retries)
        self.z3_executor = Z3Executor(timeout_s=code_timeout_s)
        self.reranker = CrossEncoderReranker()

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

        route_info = payload.get("_route_info", {})
        if route_info.get("path") == "fast":
            fast_result = self._fast_path_execute(question, premises_nl, premises_fol, payload)
            if fast_result is not None:
                fast_result.metadata["route_info"] = route_info
                return fast_result

        request_id = str(first_present(payload, ["question_id", "record_id", "id"], "")).strip()
        exact = self.exact_by_id.get(request_id) if request_id else None
        if exact is not None:
            return self._from_example(exact, confidence=0.995, source="id-exact")

        full_key = normalize_key(premises_nl, question)
        exact = self.exact_by_full_key.get(full_key)
        if exact is None and not premises_nl:
            maybe = self.exact_by_question.get(normalize_key(question))
            if maybe is not None:
                exact = maybe
        if exact is not None:
            return self._from_example(exact, confidence=0.99, source="exact")
        # Query VectorDB for context using natural language.
        # Bypassing early-return here to avoid "Context Domination" 
        # (different questions sharing the same premises have near-identical vectors).
        query_text = "\n".join([*premises_nl, question])
        orchestration_data = self._orchestrate_query(question)
        complexity = orchestration_data.get("complexity_score", 3)
        vector_k = max(5, int(complexity * 10))
        rerank_k = max(2, int(complexity * 2))
        
        # Build expanded semantic search query
        search_query = question
        anchors = orchestration_data.get("semantic_anchors", [])
        if anchors:
            search_query += "\n" + "\n".join(anchors)
            
        hits = self.index.search(
            search_query, 
            k=vector_k, 
            reranker=self.reranker, 
            rerank_top_k=rerank_k
        )

        # RAG-Based FOL Cache (Offline Pre-computation):
        # If the user provides premises_nl but NO premises_fol, we search the retrieved 
        # VectorDB hits to see if these premises were already translated during Offline Ingestion.
        if premises_nl and not premises_fol:
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
                premises_fol = list(premises_nl)

        llm_result = self._answer_with_llm(question, premises_nl, premises_fol, hits)
        if llm_result is not None:
            if llm_result.metadata.get("executor") == "z3" and not llm_result.metadata.get("execution_errors"):
                # Feedback loop if successful
                code = llm_result.metadata.get("executed_code", "")
                if code:
                    extract_and_write_back_logic(code, self.logic_knowledge_index)
            llm_result.metadata["route_info"] = route_info
            return llm_result

        if hits and hits[0].score >= self.low_match_threshold:
            return self._from_example(hits[0].item, confidence=min(0.58, hits[0].score), source="retrieval-fallback", hit=hits[0])

        return PipelineResult(
            answer="Uncertain",
            explanation="The premises do not provide enough machine-verifiable support for a definite answer.",
            cot=["Classified the query as Type 1.", "Attempted exact lookup, retrieval, and optional local LLM."],
            premises=premises_nl,
            fol="\n".join(premises_fol) if premises_fol else None,
            confidence=0.2,
            query_type="type1",
            source="fallback",
            metadata={"route_info": route_info}
        )

    def _translate_nl_to_fol(self, premises_nl: List[str]) -> List[str]:
        if not self.llm.enabled:
            return []
            
        # Lower thresholds to create more batches for parallel processing
        batches = split_into_elastic_batches(premises_nl, base_threshold=300, upper_threshold=400)
        all_fol = []
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def _process_batch(batch: str) -> List[str]:
            batch_sentences = [s.strip() for s in batch.split(".") if s.strip()]
            user_prompt = get_user_prompt(format_premises_for_prompt(batch_sentences))
            try:
                raw = self.llm.chat_json(
                    system_prompt=TRANSLATION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.2,
                    max_tokens=4096
                )
                if raw and "premises_fol" in raw:
                    return raw["premises_fol"]
            except Exception as exc:
                print(f"[EXACT] FOL translation error: {exc}", flush=True)
            return []
            
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_process_batch, batch) for batch in batches]
            for future in as_completed(futures):
                all_fol.extend(future.result())
                
        return all_fol

    def _orchestrate_query(self, question: str) -> dict:
        fallback = {
            "exact_entities": [],
            "semantic_anchors": [question],
            "complexity_score": 3,
            "is_solvable": True
        }
        if not self.expansion_llm.enabled:
            return fallback
            
        system_prompt = (
            "You are an orchestration AI. Read the problem and return ONLY a valid JSON object with the following schema:\n"
            "- `exact_entities`: List[str] (Key names, named entities, named variables)\n"
            "- `semantic_anchors`: List[str] (1-2 sentences of HyDE contextual assumptions)\n"
            "- `complexity_score`: int (1-5, where 1 is a simple lookup, 5 is a complex multi-step logical deduction)\n"
            "- `is_solvable`: bool (False if it's completely nonsensical or missing required premises)"
        )
        try:
            raw = self.expansion_llm.chat_json(
                system_prompt=system_prompt, 
                user_prompt=question,
                temperature=0.0,
                max_tokens=200
            )
            if raw:
                # If we get a response, it might already be parsed by our robust parse_llm_response,
                # or it might be raw dict. We just use it directly.
                print(f"\n🚀 [GEMMA 1B ORCHESTRATION] {json.dumps(raw, ensure_ascii=False)}\n", flush=True)
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
            fol="\n".join(example.premises_fol) if example.premises_fol else None,
            confidence=confidence,
            query_type="type1",
            source=source,
            matched_id=example.question_id,
            metadata=metadata,
        )

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
        
        from exact_pipeline.llm.templates import LOGIC_NETWORKX_TEMPLATE, LOGIC_TEMPLATE
        q_lower = question.lower()
        
        # Strict structural intent matching. 
        # This prevents Decoy phrases in Yes/No questions or multi-clause Boolean questions from triggering NetworkX.
        strict_patterns = [
            r"\b(what|which)( of the following)? is the( absolute)?( logical)? strongest( possible)? conclusion\b",
            r"\b(what|which)( of the following)? conclusion is (the )?strongest\b",
            r"\bfind the fewest (premises|steps)\b",
            r"\bwhat requires the fewest (premises|steps)\b",
            r"\bwith the minimum (number of )?(premises|steps)\b",
            r"\b(what|which) is the most direct( path| inference)?\b",
            r"\b(what|which) is the (longest|shortest) (chain|path)\b",
            r"\bfind the (longest|shortest) (chain|path)\b"
        ]
        
        is_comparative = any(re.search(pattern, q_lower) for pattern in strict_patterns)

        # If FOL is available, exclusively use it to avoid attention distraction.
        # Fallback to NL only if FOL is missing or empty.
        # EXCEPTION: NetworkX comparative queries work much better with natural language.
        if premises_fol and not is_comparative:
            display_premises = "Premises-FOL:\n" + "\n".join(premises_fol)
            premises_list = list(premises_fol)
        else:
            display_premises = "Premises-NL:\n" + "\n".join(f"{i + 1}. {premise}" for i, premise in enumerate(premises_nl))
            premises_list = list(premises_nl)
            
        if is_comparative:
            system_prompt = LOGIC_NETWORKX_TEMPLATE.render(question=question, premises=premises_list)
            subgraph_text = ""
        else:
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
                raw = self.llm.chat(system_prompt=system_prompt, user_prompt=user_prompt)
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
                return self._from_logic_execution(raw, executed, premises_nl, premises_fol, attempt + 1)
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
    ) -> PipelineResult:
        raw_cot = _ensure_list(raw.get("cot", []))
        cot = raw_cot + executed.cot
        if executed.stdout:
            cot.append("Executed generated Z3/Python verifier and parsed its RESULT payload.")
        premises = merge_premises(executed.premises, _ensure_list(raw.get("premises", [])), premises_nl)
        fol = executed.fol or str(raw.get("fol", "")) or ("\n".join(premises_fol) if premises_fol else None)
        metadata = {
            "llm_attempts": attempts,
            "executor": "z3",
            "executed_code": raw.get("python_code", raw.get("code", "")),
            **executed.metadata,
        }
        return PipelineResult(
            answer=executed.answer or str(raw.get("answer", "Uncertain")),
            explanation=executed.explanation
            or str(raw.get("explanation", ""))
            or "The generated verifier answered using the supplied logical premises.",
            cot=cot,
            premises=premises,
            fol=fol,
            confidence=float(raw.get("confidence", 0.68) or 0.68),
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
            answer=str(raw.get("answer", "Uncertain")),
            explanation=str(raw.get("explanation", "")) or "The local LLM answered using the supplied logical premises.",
            cot=_ensure_list(raw.get("cot", [])),
            premises=merge_premises(_ensure_list(raw.get("premises", [])), premises_nl),
            fol=str(raw.get("fol", "")) or ("\n".join(premises_fol) if premises_fol else None),
            confidence=min(float(raw.get("confidence", 0.52) or 0.52), 0.62),
            query_type="type1",
            source="self-hosted-llm-fallback",
            metadata=metadata,
        )


def _ensure_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


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
