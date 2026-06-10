"""Text normalization, tokenization, and answer formatting helpers."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable, List, Sequence


_PUNCT_RE = re.compile(r"[^a-z0-9_+\-./]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    raw = "" if text is None else str(text)
    raw = raw.replace("\u2212", "-")
    raw = raw.replace("\u00d7", "x")
    raw = raw.replace("\u03bc", "u").replace("\u00b5", "u")
    raw = raw.replace("\u03a9", "ohm").replace("\u2126", "ohm")
    raw = unicodedata.normalize("NFKD", raw)
    raw = raw.encode("ascii", "ignore").decode("ascii")
    raw = raw.lower()
    raw = _PUNCT_RE.sub(" ", raw)
    return _SPACE_RE.sub(" ", raw).strip()


def normalize_key(*parts: object) -> str:
    return normalize_text(" ".join(_flatten(parts)))


def _flatten(parts: Iterable[object]) -> List[str]:
    out: List[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple)):
            out.extend(_flatten(part))
        else:
            out.append(str(part))
    return out


def rough_stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: object) -> List[str]:
    return [rough_stem(tok) for tok in normalize_text(text).split() if len(tok) > 1]


def split_steps(cot: str) -> List[str]:
    if not cot:
        return []
    lines = [line.strip() for line in str(cot).splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    chunks = re.split(r"(?=(?:Step\s+\d+|[0-9]+\.)\s*:?)", cot)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def format_number(value: float, max_decimals: int = 6) -> str:
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e6:
        text = f"{value:.6e}"
        mantissa, exponent = text.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa} x 10^{int(exponent)}"
    text = f"{value:.{max_decimals}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def join_answer(answer: str, unit: str = "") -> str:
    unit = (unit or "").strip()
    if unit in {"-", "--", "\u2014"}:
        unit = ""
    return f"{answer} {unit}".strip()


def safe_float_equal(left: str, right: str, tolerance: float = 1e-6) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return normalize_text(left) == normalize_text(right)


def first_present(payload: dict, names: Sequence[str], default=None):
    for name in names:
        if name in payload:
            return payload[name]
    return default
