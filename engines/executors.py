"""Deterministic execution helpers for LLM-generated solver code.

The competition allows visible tool use. These executors keep that tool use
small, time-bounded, and easy to report in the final JSON response. They are
designed as a local fallback for development; production Docker deployment can
run the whole API in an isolated container and install ``z3-solver``/``sympy``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from exact_pipeline.utils.text_utils import join_answer, split_steps


@dataclass
class ExecutionResult:
    ok: bool
    answer: Optional[str] = None
    unit: Optional[str] = None
    explanation: Optional[str] = None
    cot: List[str] = field(default_factory=list)
    premises: List[str] = field(default_factory=list)
    fol: Optional[str] = None
    stdout: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeExecutionError(RuntimeError):
    pass


class PythonSandboxExecutor:
    """Run compact generated Python in a restricted child Python process.

    This is intentionally conservative: no filesystem, network, subprocess, or
    dynamic import helpers are exposed. The generated script should set a
    ``RESULT`` dictionary or print a JSON object with ``answer`` and optional
    explanation fields.
    """

    DEFAULT_ALLOWED_IMPORTS = frozenset({"math", "cmath", "decimal", "fractions", "statistics", "json", "sympy"})
    DISALLOWED_NAMES = frozenset(
        {
            "__import__",
            "breakpoint",
            "compile",
            "eval",
            "exec",
            "globals",
            "input",
            "locals",
            "open",
            "vars",
        }
    )
    DISALLOWED_IMPORTS = frozenset(
        {
            "builtins",
            "ctypes",
            "importlib",
            "marshal",
            "multiprocessing",
            "os",
            "pathlib",
            "pickle",
            "posix",
            "requests",
            "shutil",
            "socket",
            "subprocess",
            "sys",
            "urllib",
        }
    )

    def __init__(self, *, timeout_s: float = 4.0, allowed_imports: Optional[Iterable[str]] = None) -> None:
        self.timeout_s = timeout_s
        self.allowed_imports = frozenset(allowed_imports or self.DEFAULT_ALLOWED_IMPORTS)

    def run(self, code: str) -> ExecutionResult:
        code = clean_code_block(code)
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        if not code.strip():
            return ExecutionResult(False, error="empty_code", metadata={"code_sha256": digest})
        try:
            tree = ast.parse(code, mode="exec")
            self._validate_ast(tree)
        except Exception as exc:
            return ExecutionResult(False, error=f"syntax_or_policy_error: {exc}", metadata={"code_sha256": digest})

        try:
            completed = subprocess.run(
                [sys.executable, "-c", _EXEC_WRAPPER],
                input=json.dumps({"code": code, "allowed_imports": sorted(self.allowed_imports)}),
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
                env={"PYTHONDONTWRITEBYTECODE": "1"},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                False,
                error=f"execution_timeout:{self.timeout_s}s",
                metadata={"code_sha256": digest},
            )

        decoded = _decode_subprocess_result(completed.stdout)
        if decoded is None:
            stderr = completed.stderr.strip()
            return ExecutionResult(
                False,
                stdout=completed.stdout.strip(),
                error=f"executor_protocol_error: {stderr or 'missing result marker'}",
                metadata={"code_sha256": digest, "returncode": completed.returncode},
            )
        if not decoded.get("ok"):
            return ExecutionResult(
                False,
                stdout=str(decoded.get("stdout", "")).strip(),
                error=f"runtime_error: {decoded.get('error', 'unknown_error')}",
                metadata={"code_sha256": digest, "returncode": completed.returncode},
            )
        stdout = str(decoded.get("stdout", "")).strip()
        payload = decoded.get("payload")
        result = _coerce_payload(payload, stdout=stdout, code_digest=digest)
        if result is None:
            return ExecutionResult(
                False,
                stdout=stdout,
                error="missing_result: set RESULT dict or print a JSON object",
                metadata={"code_sha256": digest},
            )
        return result

    def _validate_ast(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    module_name = node.module if isinstance(node, ast.ImportFrom) else alias.name
                    root = (module_name or alias.name).split(".")[0]
                    if root in self.DISALLOWED_IMPORTS or root not in self.allowed_imports:
                        raise CodeExecutionError(f"import_not_allowed:{root}")
            elif isinstance(node, ast.Name) and node.id in self.DISALLOWED_NAMES:
                raise CodeExecutionError(f"name_not_allowed:{node.id}")
            elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise CodeExecutionError("dunder_attribute_not_allowed")


class Z3Executor(PythonSandboxExecutor):
    """Executor for generated Python/Z3 proof scripts."""

    def __init__(self, *, timeout_s: float = 4.0) -> None:
        super().__init__(timeout_s=timeout_s, allowed_imports={*self.DEFAULT_ALLOWED_IMPORTS, "z3", "networkx"})

    def run(self, code: str) -> ExecutionResult:
        result = super().run(code)
        if not result.ok and "No module named 'z3'" in result.error:
            result.error = "z3_unavailable: install z3-solver or rely on deterministic Horn/fallback reasoning"
        return result


def clean_code_block(code: object) -> str:
    text = "" if code is None else str(code)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python|py|z3)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


_RESULT_MARKER = "__EXACT_EXEC_RESULT__="
_EXEC_WRAPPER = r'''
import ast
import contextlib
import io
import json
import math
import re
import sys
import traceback

CONFIG = json.loads(sys.stdin.read())
CODE = CONFIG.get("code", "")
ALLOWED_IMPORTS = set(CONFIG.get("allowed_imports", []))
DISALLOWED_IMPORTS = {
    "builtins", "ctypes", "importlib", "marshal", "multiprocessing", "os",
    "pathlib", "pickle", "posix", "requests", "shutil", "socket",
    "subprocess", "sys", "urllib",
}
DISALLOWED_NAMES = {
    "__import__", "breakpoint", "compile", "eval", "exec", "globals",
    "input", "locals", "open", "vars",
}


def fail(message, stdout=""):
    print("__EXACT_EXEC_RESULT__=" + json.dumps({"ok": False, "error": message, "stdout": stdout}, default=str))


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in DISALLOWED_IMPORTS or root not in ALLOWED_IMPORTS:
        raise ImportError(f"import_not_allowed:{root}")
    return __import__(name, globals, locals, fromlist, level)


def validate(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                module_name = node.module if isinstance(node, ast.ImportFrom) else alias.name
                root = (module_name or alias.name).split(".")[0]
                if root in DISALLOWED_IMPORTS or root not in ALLOWED_IMPORTS:
                    raise RuntimeError(f"import_not_allowed:{root}")
        elif isinstance(node, ast.Name) and node.id in DISALLOWED_NAMES:
            raise RuntimeError(f"name_not_allowed:{node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise RuntimeError("dunder_attribute_not_allowed")


def json_from_stdout(stdout):
    match = re.search(r"\{.*\}", stdout, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


safe_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "pow": pow, "print": print,
    "range": range, "round": round, "set": set, "str": str, "sum": sum,
    "tuple": tuple, "zip": zip, "Exception": Exception, "ValueError": ValueError,
    "chr": chr, "ord": ord,
    "__import__": safe_import,
}
namespace = {"__builtins__": safe_builtins, "math": math}
stdout_buffer = io.StringIO()
try:
    # Pre-patch z3 >> operator to work as Implies (common LLM mistake)
    try:
        import z3 as _z3_patch
        _z3_patch.BoolRef.__rshift__ = lambda self, other: _z3_patch.Implies(self, other)
    except ImportError:
        pass
    tree = ast.parse(CODE, mode="exec")
    validate(tree)
    with contextlib.redirect_stdout(stdout_buffer):
        exec(compile(tree, "<exact-generated-code>", "exec"), namespace, namespace)
    stdout = stdout_buffer.getvalue().strip()
    payload = namespace.get("RESULT", namespace.get("result"))
    if payload is None:
        payload = json_from_stdout(stdout)
    print("__EXACT_EXEC_RESULT__=" + json.dumps({"ok": True, "payload": payload, "stdout": stdout}, default=str))
except Exception as exc:
    stdout = stdout_buffer.getvalue().strip()
    message = traceback.format_exception_only(type(exc), exc)[-1].strip()
    fail(message, stdout=stdout)
'''


def _decode_subprocess_result(stdout: str) -> Optional[Dict[str, Any]]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            try:
                decoded = json.loads(line[len(_RESULT_MARKER) :])
            except json.JSONDecodeError:
                return None
            return decoded if isinstance(decoded, dict) else None
    return None


def _coerce_payload(payload: Any, *, stdout: str, code_digest: str) -> Optional[ExecutionResult]:
    if payload is None:
        return None

    if not isinstance(payload, dict):
        payload = {"answer": payload}

    answer = _first_present(payload, ["answer", "final_answer", "result", "value"])
    unit = _first_present(payload, ["unit", "units"], "")
    if answer is None:
        answer_text = None
    else:
        answer_text = str(answer)

    cot = payload.get("cot", payload.get("steps", []))
    premises = payload.get("premises", payload.get("used_premises", []))
    result = ExecutionResult(
        ok=bool(answer_text or payload.get("explanation")),
        answer=answer_text,
        unit=str(unit) if unit else None,
        explanation=str(payload.get("explanation", "")) or None,
        cot=_string_list(cot),
        premises=_string_list(premises),
        fol=str(payload.get("fol", "")) or None,
        stdout=stdout,
        metadata={
            "code_sha256": code_digest,
            "executor_result_keys": sorted(str(key) for key in payload.keys()),
        },
    )
    return result


def _first_present(payload: Dict[str, Any], names: Iterable[str], default=None):
    for name in names:
        if name in payload and payload[name] not in (None, ""):
            return payload[name]
    return default


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    return split_steps(str(value))
