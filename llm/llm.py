"""Optional OpenAI-compatible client for a self-hosted small LLM.

Handles DeepSeek-R1's tendency to return free-form Markdown/LaTeX instead of
JSON, including <think> blocks and \\boxed{} answers.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    pass


class OpenAICompatibleLLM:
    """Call a self-hosted OpenAI-compatible chat endpoint.

    The competition Q&A disallows third-party inference APIs. Configure this
    only for infrastructure you control, for example vLLM serving an eligible
    open-source model.
    """

    def __init__(self, base_url: str = "", model: str = "", timeout_s: float = 45.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or ""
        self.timeout_s = timeout_s

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            decoded = json.loads(raw)
            content = decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError("LLM response did not match OpenAI chat format") from exc

        return parse_llm_response(content)

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        json_schema: Optional[dict] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        # Build response_format based on whether a schema is provided
        if json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": json_schema,
                }
            }
        else:
            response_format = {"type": "json_object"}

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            decoded = json.loads(raw)
            content = decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError("LLM response did not match OpenAI chat format") from exc

        return parse_llm_response(content)


# ---------------------------------------------------------------------------
# Robust response parser: JSON → structured text → Markdown/LaTeX
# ---------------------------------------------------------------------------

def parse_llm_response(text: str) -> Dict[str, Any]:
    """Parse LLM output into a dict, handling JSON, plain text, and LaTeX."""
    text = text.strip()

    # 1. Strip DeepSeek-R1 <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Try direct JSON parse
    result = _try_json_parse(text)
    if result is not None:
        return result

    # 3. Try extracting a JSON code block (```json ... ```)
    result = _try_json_codeblock(text)
    if result is not None:
        return result

    # 4. Parse free-form Markdown/LaTeX response (DeepSeek-R1 style)
    result = _parse_freeform_response(text)
    if result is not None:
        return result

    # 5. Try finding a balanced JSON object in the text
    result = _try_find_json_object(text)
    if result is not None:
        return result

    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to parse LLM response. Raw text was: {repr(text)}")
    raise LLMError("Could not parse LLM response into structured data")


def _try_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """Try parsing the entire text as a JSON object."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _try_json_codeblock(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from a ```json ... ``` code block."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, flags=re.DOTALL)
    if match:
        return _try_json_parse(match.group(1).strip())
    return None


def _try_find_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Find a balanced { ... } JSON object in the text using brace counting."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                result = _try_json_parse(candidate)
                if result is not None:
                    return result
                # If this balanced block didn't parse, try the next one
                next_start = text.find("{", i + 1)
                if next_start != -1:
                    return _try_find_json_object(text[next_start:])
                return None
    return None


def _parse_freeform_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse a free-form Markdown/LaTeX response into a structured dict.

    Handles DeepSeek-R1's typical output format:
    - Step-by-step solutions with **Step N:** markers
    - Final answers in \\boxed{...} or "Final Answer:" or "Answer:" lines
    - LaTeX formulas with \\[ ... \\] blocks
    """
    result: Dict[str, Any] = {}

    # --- Extract Python code if present ---
    python_match = re.search(r"```python\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if python_match:
        result["python_code"] = python_match.group(1).strip()
    else:
        # Fallback: if backticks are missing or unclosed, grab everything under **Python Code:**
        fallback_match = re.search(r"\*\*Python Code:\*\*\s*(.+)", text, flags=re.DOTALL | re.IGNORECASE)
        if fallback_match:
            code = fallback_match.group(1).strip()
            # Strip unclosed backticks if any
            code = re.sub(r"```[a-zA-Z]*\n?|```", "", code).strip()
            result["python_code"] = code

    # --- Extract answer ---
    answer = _extract_answer(text)
    if not answer and "python_code" not in result:
        return None  # If we can't find an answer and there is no code, give up
    
    if answer:
        result["answer"] = _clean_answer(answer)

    # --- Extract chain-of-thought steps ---
    cot = _extract_steps(text)
    if cot:
        result["cot"] = cot

    # --- Build explanation from the full text ---
    result["explanation"] = _extract_explanation(text)

    # --- Extract confidence if mentioned ---
    conf_match = re.search(r"(?i)confidence[:\s]+(\d+(?:\.\d+)?)\s*%?", text)
    if conf_match:
        val = float(conf_match.group(1))
        result["confidence"] = val / 100.0 if val > 1.0 else val
    else:
        result["confidence"] = 0.72  # Reasonable default for free-form answers



    return result


def _extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from various formats."""
    # 1. \\boxed{...} (LaTeX) - most specific, highest priority
    #    Use brace counting to handle nested braces like \boxed{20\,\text{mJ}}
    boxed_start = text.find("\\boxed{")
    if boxed_start != -1:
        content_start = boxed_start + len("\\boxed{")
        depth = 1
        i = content_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            answer = text[content_start : i - 1]
            # Clean LaTeX formatting
            answer = re.sub(r"\\,", " ", answer)
            answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
            answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
            answer = re.sub(r"\\(?:times|cdot)", "×", answer)
            answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", answer)
            answer = answer.strip()
            return answer

    # 2. "Final Answer:" line
    final_match = re.search(r"(?i)\*{0,2}Final\s+Answer\*{0,2}[:\s]+(.+?)(?:\n|$)", text)
    if final_match:
        return _clean_latex(final_match.group(1).strip())

    # 3. "Answer:" line
    answer_match = re.search(r"(?i)^\*{0,2}Answer\*{0,2}[:\s]+(.+?)$", text, re.MULTILINE)
    if answer_match:
        return _clean_latex(answer_match.group(1).strip())

    # 4. "The answer is ..." — match to end of line to avoid periods in decimals
    is_match = re.search(r"(?i)the\s+(?:final\s+)?answer\s+is\s+(.+?)$", text, re.MULTILINE)
    if is_match:
        return _clean_latex(is_match.group(1).strip().rstrip('.'))

    # 5. Last "= value unit" pattern (e.g. "E = 0.02 J")
    eq_match = re.findall(r"=\s*([\d.,]+\s*(?:[a-zA-Zμ]+(?:/[a-zA-Z]+)?))", text)
    if eq_match:
        return _clean_latex(eq_match[-1].strip())

    # 6. For logic: Yes/No/Uncertain as standalone answer
    yn_match = re.search(r"(?i)\b(Yes|No|Uncertain)\b", text)
    if yn_match:
        return yn_match.group(1).capitalize()

    return None


def _extract_steps(text: str) -> List[str]:
    """Extract step-by-step reasoning from the response."""
    steps: List[str] = []

    # Pattern 1: **Step N:** ...
    step_matches = re.findall(r"\*{0,2}(Step\s+\d+)\*{0,2}[:\s]+(.+?)(?=\n\n|\n\*{0,2}Step|\Z)", text, re.DOTALL)
    if step_matches:
        for step_label, s in step_matches:
            clean = _clean_latex(s.strip().lstrip('*').strip())
            if clean:
                steps.append(f"{step_label}: {clean}")
        return steps

    # Pattern 2: Numbered list: 1. ... 2. ...
    numbered = re.findall(r"^\s*(\d+)[\.\)]\s+(.+?)$", text, re.MULTILINE)
    if len(numbered) >= 2:
        return [f"Step {num}: {_clean_latex(s.strip())}" for num, s in numbered if s.strip()]

    return steps


def _extract_explanation(text: str) -> str:
    """Build a concise explanation from the response text."""
    # Try to find an explicit explanation section
    exp_match = re.search(r"(?i)\*?\*?Explanation\*?\*?[:\s]+(.+?)(?=\n\n\*?\*?[A-Z]|\Z)", text, re.DOTALL)
    if exp_match:
        return _clean_latex(exp_match.group(1).strip())

    # Otherwise use the first meaningful paragraph
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for p in paragraphs:
        clean = _clean_latex(p)
        if len(clean) > 30 and not clean.startswith("```"):
            return clean[:500]

    return _clean_latex(text[:300])


def _clean_latex(text: str) -> str:
    """Remove LaTeX markup for readable plain text."""
    text = re.sub(r"\\\[(.+?)\\\]", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", r"\1", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:times|cdot)", "×", text)
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\,", " ", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = text.strip()
    return text


def _clean_answer(answer: str) -> str:
    """Post-process an extracted answer string."""
    # Strip common prefixes: "is ", "= ", "E = ", etc.
    answer = re.sub(r"^(?:is\s+)", "", answer, flags=re.IGNORECASE).strip()
    # Strip trailing periods (but not if it looks like "0.02")
    if answer.endswith(".") and not re.search(r"\d\.$", answer):
        answer = answer.rstrip(".")
    # Strip variable assignment prefix: "E = 0.02 J" → "0.02 J"
    var_match = re.match(r"^[A-Za-z_]\w*\s*=\s*(.+)$", answer)
    if var_match:
        answer = var_match.group(1).strip()
    return answer
