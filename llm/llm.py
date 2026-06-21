"""Optional OpenAI-compatible client for a self-hosted small LLM.

Handles DeepSeek-R1's tendency to return free-form Markdown/LaTeX instead of
JSON, including <think> blocks and \\boxed{} answers.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import socket
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    pass


class OpenAICompatibleLLM:
    """Call a self-hosted OpenAI-compatible chat endpoint.

    The competition Q&A disallows third-party inference APIs. Configure this
    only for infrastructure you control, for example vLLM serving an eligible
    open-source model.
    """

    def __init__(self, base_url: str = "", model: str = "", timeout_s: float = 300.0) -> None:
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
        thinking: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        system_prompt, user_prompt = _apply_qwen_thinking_mode(
            system_prompt, user_prompt, thinking=thinking
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(_qwen_hard_thinking_payload(self.model, thinking=False))
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
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            decoded = json.loads(raw)
            content = _extract_openai_message_content(decoded)
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
        thinking: bool = False,
        request_timeout_s: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        system_prompt, user_prompt = _apply_qwen_thinking_mode(
            system_prompt, user_prompt, thinking=thinking
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(_qwen_hard_thinking_payload(self.model, thinking=False))
        payload.update(_structured_output_payload(json_schema))
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            timeout_s = self.timeout_s if request_timeout_s is None else min(
                self.timeout_s, max(0.1, float(request_timeout_s))
            )
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            decoded = json.loads(raw)
            content = _extract_openai_message_content(decoded)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError("LLM response did not match OpenAI chat format") from exc

        return parse_llm_response(content)


# ---------------------------------------------------------------------------
# Robust response parser: JSON → structured text → Markdown/LaTeX
# ---------------------------------------------------------------------------

def _extract_openai_message_content(decoded: Dict[str, Any]) -> str:
    message = decoded["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", part.get("content", "")) if isinstance(part, dict) else part)
            for part in content
        )
    if content:
        return str(content)
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return str(reasoning or "")


def _apply_qwen_thinking_mode(
    system_prompt: str, user_prompt: str, *, thinking: bool
) -> tuple[str, str]:
    """Force Qwen3 non-thinking mode while preserving the public client API."""

    mode_hint = os.getenv("EXACT_QWEN_MODE_HINT", "auto").lower()
    if mode_hint in {"off", "false", "0", "none"}:
        return system_prompt, user_prompt

    system_prompt = re.sub(r"/(?:no_)?think\b", "", system_prompt).rstrip()
    user_prompt = re.sub(r"/(?:no_)?think\b", "", user_prompt).rstrip()
    instruction = "Use Qwen3 non-thinking mode and return only the requested structured output."
    return f"{system_prompt}\n\n{instruction} /no_think", f"{user_prompt}\n\n/no_think"


def _qwen_hard_thinking_payload(model: str, *, thinking: bool) -> Dict[str, Any]:
    """Disable Qwen3 thinking through the serving stack's hard switch."""

    mode = os.getenv("EXACT_QWEN_HARD_THINKING_SWITCH", "auto").lower()
    if mode in {"off", "false", "0", "none"}:
        return {}
    model_name = str(model or "").lower()
    should_send = mode in {"on", "true", "1"} or "qwen3" in model_name
    if not should_send:
        return {}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _structured_output_payload(json_schema: Optional[dict]) -> Dict[str, Any]:
    """Build the JSON-schema contract understood by llama.cpp and vLLM."""

    if not json_schema:
        return {}
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "exact_structured_response",
                "strict": True,
                "schema": json_schema,
            },
        }
    }


def parse_llm_response(text: str) -> Dict[str, Any]:
    """Parse LLM output into a dict, handling JSON, plain text, and LaTeX."""
    original_text = str(text or "").strip()
    text = original_text

    # 1. Strip DeepSeek-R1 <think>...</think> blocks
    without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if without_think:
        text = without_think
    elif "<think>" in text:
        text = re.sub(r"</?think>", "", original_text).strip()

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
    preview = text if len(text) <= 2400 else text[:1200] + "\n...<truncated>...\n" + text[-1200:]
    logger.error(
        "Failed to parse LLM response (%d chars). Preview: %r",
        len(text),
        preview,
    )
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
            
    if "python_code" not in result:
        # Check if they just wrote python code straight in the response
        if "RESULT" in text and "=" in text:
            # Look for RESULT = {...}
            res_match = re.search(r"RESULT\s*=\s*\{.*\}", text, flags=re.DOTALL)
            if res_match:
                result["python_code"] = res_match.group(0)

    # --- Extract answer ---
    answer = _extract_answer(text)
    if not answer and "python_code" not in result:
        return None  # If we can't find an answer and there is no code, give up
    
    if answer:
        result["answer"] = _clean_answer(answer)

    # --- Extract chain-of-thought steps ---
    cot = _extract_steps(text)
    if "answer" not in result:
        result["answer"] = "Uncertain"
    if "explanation" not in result:
        # Don't use the raw text as explanation if it's mostly code
        if "python_code" in result and len(text) - len(result["python_code"]) < 50:
            result["explanation"] = "Generated verifier code to evaluate the logic."
        else:
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

    # 7. Multiple choice standalone answer (A, B, C, D) if formatted clearly
    mc_match = re.search(r"(?i)(?:correct\s+(?:option|answer)\s+is\s+|answer:\s*)([A-D])\b", text)
    if mc_match:
        return mc_match.group(1).upper()

    # 8. Very short response fallback (e.g. just "B", "50", "Option A")
    if len(text.strip()) < 20:
        short_mc = re.search(r"(?i)^\s*(?:Option\s+)?([A-D])\s*\.?\s*$", text)
        if short_mc:
            return short_mc.group(1).upper()
        # If it's just a number
        short_num = re.search(r"^\s*([+-]?\d+(?:\.\d+)?)\s*$", text)
        if short_num:
            return short_num.group(1)
            
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
