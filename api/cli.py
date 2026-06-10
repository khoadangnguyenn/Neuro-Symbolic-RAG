"""Command-line interface for local EXACT pipeline testing."""

from __future__ import annotations

import argparse
import json
import sys

from exact_pipeline.orchestration.pipeline import ExactPipeline
from exact_pipeline.api.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="EXACT 2026 hybrid baseline pipeline")
    parser.add_argument("--serve", action="store_true", help="Run the HTTP API server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--input-json", help="Path to a JSON query payload")
    parser.add_argument("--question", help="Question/problem text for a quick single query")
    parser.add_argument("--query-type", choices=["type1", "type2"], help="Force query type")
    parser.add_argument("--sample", choices=["type1", "type2"], help="Run a built-in sample from the dataset")
    parser.add_argument("--stats", action="store_true", help="Print dataset/pipeline stats")
    args = parser.parse_args()

    if args.serve:
        run_server(host=args.host, port=args.port)
        return

    pipeline = ExactPipeline()
    if args.stats:
        print(json.dumps(pipeline.stats(), ensure_ascii=False, indent=2))
        return

    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    elif args.question:
        payload = {"question": args.question}
        if args.query_type:
            payload["query_type"] = args.query_type
    elif args.sample:
        if args.sample == "type1":
            example = pipeline.logic_examples[0]
            payload = {
                "query_type": "type1",
                "premises-NL": list(example.premises_nl),
                "question": example.question,
            }
        else:
            example = pipeline.physics_examples[0]
            payload = {"query_type": "type2", "question": example.question}
    else:
        payload = json.load(sys.stdin)

    response = pipeline.answer(payload)
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
