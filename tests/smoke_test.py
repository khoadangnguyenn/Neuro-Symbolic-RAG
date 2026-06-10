#!/usr/bin/env python3
"""Smoke tests that do not require pytest."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exact_pipeline.orchestration.pipeline import ExactPipeline  # noqa: E402


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    pipeline = ExactPipeline()

    logic_example = pipeline.logic_examples[0]
    logic_response = pipeline.answer(
        {
            "query_type": "type1",
            "premises-NL": list(logic_example.premises_nl),
            "question": logic_example.question,
        }
    )
    assert_true(logic_response["answer"] == logic_example.answer, "Type 1 exact sample failed")
    assert_true("explanation" in logic_response, "Type 1 response missing explanation")

    physics_example = pipeline.physics_examples[0]
    physics_response = pipeline.answer({"query_type": "type2", "question": physics_example.question})
    assert_true(physics_example.answer in physics_response["answer"], "Type 2 exact sample failed")
    assert_true("cot" in physics_response, "Type 2 response missing cot")

    print("smoke tests passed")


if __name__ == "__main__":
    main()
