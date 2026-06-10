import urllib.request
import json
import sys

payload = {
    "model": "exact-model",
    "messages": [
        {"role": "system", "content": """You are an expert physics solver. You MUST use Python to perform all arithmetic calculations to avoid errors.
1. **Reasoning:** Explain your physical reasoning, unit conversions, and formulas step-by-step using Markdown and LaTeX.
2. **Computation:** Write an executable ```python\n...\n``` code block strictly to compute the final numerical answer.
3. **Code Rules:** The Python code must be safe (math/sympy only, no file/network access).
4. **Data Extraction:** Inside the Python code, you MUST define a dictionary named RESULT with EXACTLY these keys: 'answer' (numeric/string), 'unit' (string), 'explanation' (string), 'cot' (list of strings), 'premises' (list of strings). Populate these string fields based on your physical reasoning from Step 1.
   - CRITICAL RULE 1: For the string values in 'explanation', 'cot', and 'premises', you MUST wrap the text in triple quotes (\"\"\"\") to prevent syntax escaping errors with LaTeX.
   - CRITICAL RULE 2: You MUST print the dictionary at the end of the code block (e.g., using print(RESULT) or print(json.dumps(RESULT))).
5. **Final Output:** Outside the code block, explicitly state the 'Final Answer:' followed by the value and units."""},
        {"role": "user", "content": """Question:
At the three vertices of right-angled triangle ABC (right-angled at A), with AB = 30 cm, AC = 40 cm, and BC = 50 cm, charges q1 = q2 = q3 = 2x10^-9 C are placed. Determine the magnitude of the net electric force acting on a charge q = -2x10^-9 C placed at point H, which is the foot of the altitude from A

Solve this problem step by step and give the final answer with units."""}
    ],
    "temperature": 0.0,
    "max_tokens": 8192
}

req = urllib.request.Request(
    "http://localhost:8001/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=300) as response:
        raw = response.read().decode("utf-8")
    decoded = json.loads(raw)
    content = decoded["choices"][0]["message"]["content"]
    print("--- RAW LLM OUTPUT ---")
    print(content)
except Exception as e:
    print("Error:", e)
