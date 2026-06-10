"""Dataclasses shared across pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class LogicExample:
    record_id: str
    question_id: str
    premises_nl: Sequence[str]
    premises_fol: Sequence[str]
    question: str
    answer: str
    explanation: str
    premise_indices: Sequence[int] = field(default_factory=list)


@dataclass(frozen=True)
class PhysicsExample:
    problem_id: str
    question: str
    cot: str
    answer: str
    unit: str


@dataclass
class PipelineResult:
    answer: str
    explanation: str
    cot: List[str] = field(default_factory=list)
    premises: List[str] = field(default_factory=list)
    fol: Optional[str] = None
    confidence: float = 0.0
    query_type: Optional[str] = None
    source: Optional[str] = None
    matched_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_response(self) -> Dict[str, Any]:
        """Return the EXACT-compatible JSON response.

        Only answer and explanation are required by the competition, but the
        extra fields are useful for the P3 reasoning-depth criterion.
        """

        response: Dict[str, Any] = {
            "answer": self.answer,
            "explanation": self.explanation,
        }
        if self.fol:
            response["fol"] = self.fol
        if self.cot:
            response["cot"] = self.cot
        if self.premises:
            response["premises"] = self.premises
        response["confidence"] = round(float(self.confidence), 4)
        if self.query_type:
            response["query_type"] = self.query_type
        if self.source:
            response["source"] = self.source
        if self.matched_id:
            response["matched_id"] = self.matched_id
        if self.metadata:
            response["metadata"] = self.metadata
        return response
