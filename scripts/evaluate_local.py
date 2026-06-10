#!/usr/bin/env python3
"""Evaluate exact-match behavior on the provided train files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exact_pipeline.orchestration.pipeline import ExactPipeline  # noqa: E402
from exact_pipeline.utils.text_utils import join_answer, normalize_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Local sanity evaluation for EXACT pipeline")
    parser.add_argument("--max", type=int, default=0, help="Limit examples per type; 0 means all")
    args = parser.parse_args()

    pipeline = ExactPipeline()
    max_items = args.max or None
    logic_total = logic_ok = 0
    for example in pipeline.logic_examples[:max_items]:
        payload = {
            "query_type": "type1",
            "question_id": example.question_id,
            "premises-NL": list(example.premises_nl),
            "question": example.question,
        }
        response = pipeline.answer(payload)
        logic_total += 1
        logic_ok += int(normalize_text(response["answer"]) == normalize_text(example.answer))

    physics_total = physics_ok = 0
    for example in pipeline.physics_examples[:max_items]:
        payload = {"query_type": "type2", "id": example.problem_id, "question": example.question}
        response = pipeline.answer(payload)
        expected = join_answer(example.answer, example.unit)
        physics_total += 1
        physics_ok += int(normalize_text(response["answer"]) == normalize_text(expected))

    print(
        json.dumps(
            {
                "logic": {"correct": logic_ok, "total": logic_total},
                "physics": {"correct": physics_ok, "total": physics_total},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
