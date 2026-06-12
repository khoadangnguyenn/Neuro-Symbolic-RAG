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
    
    # EXACT 2026 Format Required Fields
    unit: str = ""
    premises_used: List[int] = field(default_factory=list)
    reasoning: Optional[Dict[str, Any]] = None

    def to_response(self) -> Dict[str, Any]:
        """Return the EXACT 2026 compatible JSON response.
        
        The schema strictly requires exactly 6 keys:
        query_id, answer, unit, explanation, premises_used, reasoning.
        (query_id is injected by the orchestrator).
        """
        reasoning = self.reasoning
        if not reasoning and (self.fol or self.cot):
            reasoning = {
                "type": "fol" if self.fol else "cot",
                "steps": self.cot if self.cot else []
            }
            
        return {
            "answer": self.answer,
            "unit": self.unit,
            "explanation": self.explanation,
            "premises_used": self.premises_used,
            "reasoning": reasoning
        }
