"""Unified EXACT 2026 pipeline router."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from exact_pipeline.core.config import Settings
from exact_pipeline.core.data import load_logic_examples, load_physics_examples, validate_dataset_paths
from exact_pipeline.llm.llm import OpenAICompatibleLLM
from exact_pipeline.engines.logic import LogicPipeline
from exact_pipeline.core.models import PipelineResult
from exact_pipeline.engines.physics import PhysicsPipeline
from exact_pipeline.utils.text_utils import first_present, normalize_text


class ExactPipeline:
    """Single endpoint pipeline for both EXACT dataset types."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings.from_env()
        validate_dataset_paths([self.settings.logic_path, self.settings.physics_path])
        self.logic_examples = load_logic_examples(self.settings.logic_path)
        self.physics_examples = load_physics_examples(self.settings.physics_path)
        llm = OpenAICompatibleLLM(
            base_url=self.settings.llm_base_url,
            model=self.settings.llm_model,
            timeout_s=self.settings.llm_timeout_s,
        )
        expansion_llm = OpenAICompatibleLLM(
            base_url=self.settings.expansion_llm_base_url,
            model=self.settings.expansion_llm_model,
            timeout_s=self.settings.llm_timeout_s,
        )
        from exact_pipeline.orchestration.router import AdaptiveRouter
        self.router = AdaptiveRouter(self.settings)

        db_path = str(self.settings.project_root / "dataset" / "chromadb_data")
        common = {
            "db_path": db_path,
            "alpha": self.settings.alpha,
            "llm": llm,
            "expansion_llm": expansion_llm,
            "retrieval_k": self.settings.retrieval_k,
            "high_match_threshold": self.settings.high_match_threshold,
            "low_match_threshold": self.settings.low_match_threshold,
            "max_retries": self.settings.max_retries,
            "code_timeout_s": self.settings.code_timeout_s,
        }
        self.logic = LogicPipeline(
            self.logic_examples, 
            graph_path=str(self.settings.graph_dir / "logic_graph.graphml"),
            **common
        )
        self.physics = PhysicsPipeline(
            self.physics_examples,
            graph_path=str(self.settings.graph_dir / "physics_graph.graphml"),
            **common
        )

    def answer(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        # --- Multi-question support ---
        questions = payload.get("questions")
        if isinstance(questions, list) and len(questions) > 1:
            return self._answer_multi(payload, questions)

        # If "questions" has exactly 1 item, promote it to "question"
        if isinstance(questions, list) and len(questions) == 1:
            payload = {**payload, "question": questions[0]}

        result = self.answer_result(payload)
        resp = result.to_response()
        resp["query_id"] = str(first_present(payload, ["query_id", "id"], ""))
        return [resp]

    def _answer_multi(self, payload: Dict[str, Any], questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions sharing the same premises."""
        results: List[Dict[str, Any]] = []
        for i, q in enumerate(questions):
            sub_payload = {**payload, "question": q}
            # Remove the array key so sub-calls don't recurse
            sub_payload.pop("questions", None)
            try:
                result = self.answer_result(sub_payload)
                resp = result.to_response()
                resp["question_index"] = i
                resp["question_text"] = q
                results.append(resp)
            except Exception as exc:
                results.append({
                    "question_index": i,
                    "question_text": q,
                    "answer": "Uncertain",
                    "explanation": f"Pipeline error: {exc}",
                    "confidence": 0.0,
                    "source": "server-error",
                })
        return results

    def answer_result(self, payload: Dict[str, Any]) -> PipelineResult:
        query_type = infer_query_type(payload)
        question = str(first_present(payload, ["question", "query", "problem"], ""))
        
        # Adaptive Routing
        if question:
            route_info = self.router.route(question)
            payload["_route_info"] = route_info
            
        if query_type == "type1":
            return self.logic.answer(payload)
        if query_type == "type2":
            return self.physics.answer(payload)
        return PipelineResult(
            answer="Uncertain",
            explanation="The query type could not be inferred. Provide query_type='type1' or query_type='type2'.",
            confidence=0.0,
            source="router",
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "logic_examples": len(self.logic_examples),
            "physics_examples": len(self.physics_examples),
            "llm_enabled": bool(self.settings.llm_base_url and self.settings.llm_model),
            "llm_model": self.settings.llm_model or None,
        }


def infer_query_type(payload: Dict[str, Any]) -> str:
    raw_type = first_present(payload, ["query_type", "type", "dataset_type", "task_type"], "")
    normalized_type = normalize_text(raw_type)
    if normalized_type in {"type1", "1", "logic", "logic based", "logic_based", "educational", "fol"}:
        return "type1"
    if normalized_type in {"type2", "2", "physics", "physic"}:
        return "type2"

    if any(key in payload for key in ("premises-NL", "premises_nl", "premises", "premises-FOL", "premises_fol", "fol")):
        return "type1"

    # Also check first item of "questions" array for type inference
    question = str(first_present(payload, ["question", "query", "problem"], ""))
    if not question:
        questions = payload.get("questions")
        if isinstance(questions, list) and questions:
            question = str(questions[0])
    normalized_question = normalize_text(question)
    physics_markers = {
        "calculate",
        "capacitor",
        "resistor",
        "circuit",
        "voltage",
        "current",
        "charge",
        "force",
        "electric",
        "resistance",
        "capacitance",
        "energy",
        "power",
    }
    if physics_markers & set(normalized_question.split()):
        return "type2"
    return "type1"
