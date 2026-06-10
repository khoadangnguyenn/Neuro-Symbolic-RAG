"""HTTP API for the EXACT pipeline with Flask and stdlib backends."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from exact_pipeline.orchestration.pipeline import ExactPipeline


def model_payload(pipeline: ExactPipeline) -> Dict[str, Any]:
    stats = pipeline.stats()
    model_id = stats.get("llm_model") or "deterministic-exact-pipeline"
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "self-hosted" if stats.get("llm_enabled") else "deterministic-baseline",
            }
        ],
    }


def create_flask_app(pipeline: ExactPipeline | None = None):
    """Create the Flask API app when Flask is installed."""

    try:
        from flask import Flask, jsonify, request
    except ImportError as exc:  # pragma: no cover - dependency optional locally
        raise RuntimeError("Flask is not installed; use the stdlib server or install requirements.txt") from exc

    app = Flask(__name__)
    app.pipeline = pipeline or ExactPipeline()  # type: ignore[attr-defined]

    @app.get("/health")
    @app.get("/healthz")
    def health():
        return jsonify({"status": "ok", **app.pipeline.stats()})  # type: ignore[attr-defined]

    @app.get("/v1/models")
    def models():
        return jsonify(model_payload(app.pipeline))  # type: ignore[attr-defined]

    @app.post("/")
    @app.post("/answer")
    @app.post("/predict")
    @app.post("/api/answer")
    def answer():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON body must be an object"}), int(HTTPStatus.BAD_REQUEST)
        try:
            response = app.pipeline.answer(payload)  # type: ignore[attr-defined]
        except Exception as exc:  # Keep API alive during evaluation.
            response = {
                "answer": "Uncertain",
                "explanation": f"Pipeline error: {exc}",
                "confidence": 0.0,
                "source": "server-error",
            }
        return jsonify(response)

    return app


class ExactRequestHandler(BaseHTTPRequestHandler):
    pipeline: ExactPipeline

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        if self.path in {"/health", "/healthz"}:
            self._send_json({"status": "ok", **self.pipeline.stats()})
            return
        if self.path == "/v1/models":
            self._send_json(model_payload(self.pipeline))
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook
        if self.path not in {"/", "/answer", "/predict", "/api/answer"}:
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            response = self.pipeline.answer(payload)
        except Exception as exc:  # Keep API alive during evaluation.
            response = {
                "answer": "Uncertain",
                "explanation": f"Pipeline error: {exc}",
                "confidence": 0.0,
                "source": "server-error",
            }
        self._send_json(response)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> Dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length <= 0:
            raise ValueError("Empty JSON body")
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def make_server(host: str, port: int, pipeline: ExactPipeline) -> ThreadingHTTPServer:
    class Handler(ExactRequestHandler):
        pass

    Handler.pipeline = pipeline
    return ThreadingHTTPServer((host, port), Handler)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    pipeline = ExactPipeline()
    backend = os.getenv("EXACT_SERVER_BACKEND", "flask").strip().lower()
    if backend != "stdlib":
        try:
            app = create_flask_app(pipeline)
        except RuntimeError:
            pass
        else:
            print(f"EXACT Flask pipeline listening on http://{host}:{port}/answer")
            app.run(host=host, port=port, threaded=True)
            return

    server = make_server(host, port, pipeline)
    print(f"EXACT pipeline listening on http://{host}:{port}/answer")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EXACT 2026 pipeline HTTP API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
