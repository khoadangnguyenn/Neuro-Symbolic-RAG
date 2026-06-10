"""Dataset loaders for EXACT Type 1 and Type 2."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from exact_pipeline.core.models import LogicExample, PhysicsExample


def _premise_indices(raw_idx, question_position: int) -> Sequence[int]:
    if not isinstance(raw_idx, list) or question_position >= len(raw_idx):
        return []
    raw = raw_idx[question_position]
    if not isinstance(raw, list):
        return []
    values = []
    for item in raw:
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    return values


def load_logic_examples(path: Path) -> List[LogicExample]:
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    examples: List[LogicExample] = []
    for record_pos, record in enumerate(records):
        premises_nl = list(record.get("premises-NL") or [])
        premises_fol = list(record.get("premises-FOL") or [])
        questions = list(record.get("questions") or [])
        answers = list(record.get("answers") or [])
        explanations = list(record.get("explanation") or [])
        for q_pos, question in enumerate(questions):
            answer = answers[q_pos] if q_pos < len(answers) else ""
            explanation = explanations[q_pos] if q_pos < len(explanations) else ""
            examples.append(
                LogicExample(
                    record_id=f"logic-{record_pos:04d}",
                    question_id=f"logic-{record_pos:04d}-q{q_pos + 1}",
                    premises_nl=premises_nl,
                    premises_fol=premises_fol,
                    question=question,
                    answer=str(answer),
                    explanation=str(explanation),
                    premise_indices=_premise_indices(record.get("idx"), q_pos),
                )
            )
    return examples


def load_physics_examples(path: Path, filter_qa_prefix: bool = True) -> List[PhysicsExample]:
    examples: List[PhysicsExample] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            problem_id = (row.get("id") or "").strip()
            if filter_qa_prefix and problem_id.upper().startswith("QA"):
                continue
            examples.append(
                PhysicsExample(
                    problem_id=problem_id,
                    question=(row.get("question") or "").strip(),
                    cot=(row.get("cot") or "").strip(),
                    answer=(row.get("answer") or "").strip(),
                    unit=(row.get("unit") or "").strip(),
                )
            )
    return examples


def logic_document(example: LogicExample) -> str:
    return "\n".join(
        [
            *example.premises_nl,
            *example.premises_fol,
            example.question,
            example.answer,
            example.explanation,
        ]
    )


def physics_document(example: PhysicsExample) -> str:
    return "\n".join([example.question, example.cot, example.answer, example.unit])


def validate_dataset_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset file(s): " + ", ".join(missing))
