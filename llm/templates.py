"""Proknow-RAG Jinja2 Templates."""

from jinja2 import Template

PHYSICS_TEMPLATE = Template("""
You are an expert physicist and Python programmer. Solve the physics problem using the sympy library.

IMPORTANT RULES:
- Use sympy.symbols() for unknowns.
- Use sympy.Eq() for equations and sympy.solve() to solve.
- CRITICAL RULE (Coordinate Guardrail): For ANY spatial, 2D, 3D, or geometric problem (e.g. points, distances, electric/magnetic forces, vectors), you MUST assign explicit (x, y) or (x, y, z) coordinates to ALL points. 
- If a point's coordinates are unknown but its distances to other known points are given, you MUST use `sympy.solve` to find its exact (x, y) coordinates first. Set up equations like `sympy.Eq((x - x1)**2 + (y - y1)**2, d1**2)`.
- NEVER attempt to guess distances or add/subtract vector magnitudes directly. You must compute distances using the distance formula: `sqrt((x2-x1)**2 + (y2-y1)**2)`.
- Always decompose vectors into their X and Y components to find the net vector before taking the magnitude.
- You MUST set a global variable RESULT = {"answer": "...", "explanation": "..."}.
- CRITICAL: If options are provided (A, B, C, D), your `RESULT["answer"]` MUST BE ONLY THE SINGLE LETTER of the correct option (e.g. "A").
- IMPORTANT: Use `float()` instead of `.evalf()` for the final answer to avoid AttributeError when standard Python floats are generated (unless the answer is a letter A/B/C/D).
- Do NOT use print(). Only set RESULT.
- The "answer" value must be JUST the numerical value (e.g. "2.5", "100") or the option letter (e.g. "A").
- You MUST provide the unit separately in the "unit" key (e.g. "J", "V").
- The Retrieved Formulas/Rules are already pre-compiled or structured. DO NOT re-translate them. Just use them directly to translate the User Question into the final solver code.
- Output ONLY valid Python/sympy code enclosed in python blocks. NO explanations, NO pleasantries, NO markdown outside the code block.

Here is a complete working example of the Coordinate Guardrail:

**Example Problem:** "Two point charges q1=5C and q2=-5C are located 3m apart. Find force on q3=1C which is 4m from q1 and 5m from q2."
**Example Code:**
```python
import sympy
k = 9e9
q1, q2, q3 = 5, -5, 1
# 1. Assign Known Coordinates
p1 = (0, 0)
p2 = (3, 0)
# 2. Solve for Unknown Coordinates (p3)
x, y = sympy.symbols('x y', real=True)
eq1 = sympy.Eq(x**2 + y**2, 4**2)
eq2 = sympy.Eq((x - 3)**2 + y**2, 5**2)
sol = sympy.solve([eq1, eq2], (x, y))
# Pick the first valid solution (e.g. positive y)
p3 = [s for s in sol if s[1] >= 0][0]

# Distances (re-calculate or use known)
r13 = 4
r23 = 5
# Signed Magnitudes (positive=repulsive, negative=attractive)
F13 = k * (q1*q3) / r13**2
F23 = k * (q2*q3) / r23**2
# Unit Vectors (Target - Source, pointing outward)
u13 = ((p3[0]-p1[0])/r13, (p3[1]-p1[1])/r13)
u23 = ((p3[0]-p2[0])/r23, (p3[1]-p2[1])/r23)
# Force components (Add them up, signs handle direction automatically)
Fx = F13 * u13[0] + F23 * u23[0]
Fy = F13 * u13[1] + F23 * u23[1]
net_F = sympy.sqrt(Fx**2 + Fy**2)
RESULT = {"answer": str(float(net_F)), "unit": "N", "explanation": "Calculated net force using vector addition."}
```

Now solve this problem:

Problem: {{ question }}

Retrieved Formulas / Rules:
{% for premise in premises %}
- {{ premise }}
{% endfor %}

**Python Code:**
```python
(your sympy code that sets RESULT)
```
""")

LOGIC_TEMPLATE = Template("""
You are an expert logician and Python programmer. Solve the logic problem using the z3-solver.

IMPORTANT RULES:
- Use `z3` (z3-solver).
- Use `Function('Name', Object, BoolSort())` for predicates that take arguments (e.g. `C(x)`, `M(x)`). Do NOT use `Bool('Name', x)` or `Bool('Name')` for predicates with arguments!
- Use `Bool('Name')` ONLY for simple propositions that do NOT take arguments (e.g. "It is raining").
- You MUST explicitly declare `x = Const('x', Object)` before using it in any `ForAll(x, ...)`. This prevents NameError. NEVER use the Sort `Object` itself as a variable! For example, `ForAll(Object, ...)` is completely invalid.
- CRITICAL: Read through the ENTIRE 'Premises:' list. Every single predicate (e.g. `completed_ethics_training`, `has_supervisor_approval`) MUST be declared as a `Function('...', Object, BoolSort())` at the top of the script. If you forget to declare one, you will get a NameError! Do not skip any!
- For ALL predicates used in the problem AND the multiple-choice options, you MUST declare them. Do NOT forget predicates that only appear in the options!
- For First-Order Logic, declare a sort and the function: `Object = DeclareSort('Object')`, `P = Function('P', Object, BoolSort())`, `x = Const('x', Object)`.
- When translating universal statements ('If a drone...', 'Every drone...'), you MUST strictly wrap the implications in ForAll(x, ...). For example: `ForAll(x, Implies(Not(G(x)), Not(H(x))))`.
- For logical negation, ALWAYS use `Not(...)`. NEVER use the bitwise `~` operator (e.g. do not use `~G(x)`).
- Add ALL premises as constraints with s.add(). For implication, ALWAYS use Implies(a, b). NEVER use the >> operator.
- If the user provides multiple-choice options (A, B, C, D), DO NOT just check if the premises are SAT. You MUST iterate through each option to prove Entailment. To prove an option is correct, you must push the NEGATION of the option into the solver (`s.add(Not(option))`). If the result is `unsat`, that option is the correct logical conclusion. Use `s.push()` and `s.pop()` for each option.
- CRITICAL: If options are provided (A, B, C, D), your `correct_answer` MUST BE EXACTLY THE SINGLE LETTER of the correct option (e.g. "A"). Do NOT assign text like "Yes" or "No" to `correct_answer` if they correspond to option letters. Double check that you are assigning "A" when checking Option A, and not swapping letters.
- For Yes/No/Uncertain questions WITHOUT lettered options, test the target condition. If `s.add(Not(Target))` is `unsat`, `answer` is "Yes". Else if `s.add(Target)` is `unsat`, `answer` is "No". Else, `answer` is "Uncertain". Do NOT do this if options (A, B) are provided!
- Do NOT invent new predicate names for concepts already in the glossary! If the glossary/premise uses a specific name like `ground_gets_wet`, you MUST use exactly that name for the options. However, if an option introduces a COMPLETELY NEW concept (e.g. 'It is snowing' when premises only mention rain), you MUST declare a new predicate for it (e.g. `snowing = Function('snowing', Object, BoolSort())`). Do NOT reuse existing predicates for unrelated concepts.
- Use `IntSort()`, `RealSort()`, or `BoolSort()` for predicates/variables when appropriate.
- You MUST declare constants for entities AND numeric literals using `Const('Name', Object)` instead of passing python strings or ints like `'Asha'` or `12` directly to Z3 functions. For example: `Asha = Const('Asha', Object)` and then `f(Asha)`. For numbers: `num_12 = Const('12', Object)` and then `f(num_12)`. DO NOT pass python `12` to a Function expecting `Object`.
- For open-ended or numeric questions ("How many...", "What is..."), if Z3 cannot compute it algebraically, you may bypass Z3 checking and extract the answer directly from the premises.
- You MUST set a global variable RESULT = {"answer": "...", "explanation": "..."}.
- Do NOT use print(). Only set RESULT.
- The retrieved Premises/Subgraph are already pre-compiled Z3 rules or extracted facts. DO NOT re-translate them into FOL step-by-step. Just incorporate them and translate the final User Question into an `s.add(...)` or graph condition.
- Output ONLY valid Python/Z3 code enclosed in python blocks. NO explanations, NO pleasantries, NO markdown outside the code block.

Here is a complete working example for Z3 (for standard queries):

**Example Problem:** "All dogs are animals. Rex is a dog. Which is true? A) Rex is not a dog B) Rex is an animal C) All animals are dogs"
**Example Code:**
```python
from z3 import *
s = Solver()
Object = DeclareSort('Object')
Dog = Function('Dog', Object, BoolSort())
Animal = Function('Animal', Object, BoolSort())
rex = Const('rex', Object)
x = Const('x', Object)

# Premise: All dogs are animals
s.add(ForAll(x, Implies(Dog(x), Animal(x))))
# Premise: Rex is a dog
s.add(Dog(rex))

# Entailment Check for Options
correct_answer = None

# Test Option A: Rex is not a dog
s.push()
option_A = Not(Dog(rex))
s.add(Not(option_A))
if s.check() == unsat:
    correct_answer = "A"
s.pop()

# Test Option B: Rex is an animal
s.push()
option_B = Animal(rex)
s.add(Not(option_B))
if s.check() == unsat:
    correct_answer = "B"
s.pop()

# Test Option C: All animals are dogs
s.push()
option_C = ForAll(x, Implies(Animal(x), Dog(x)))
s.add(Not(option_C))
if s.check() == unsat:
    correct_answer = "C"
s.pop()

if correct_answer:
    RESULT = {"answer": correct_answer, "explanation": f"Option {correct_answer} logically follows from the premises because its negation leads to a contradiction (UNSAT)."}
else:
    RESULT = {"answer": "Uncertain", "explanation": "No option logically follows from the premises."}
```

**Example Problem 2 (Yes/No/Uncertain):** "All dogs are animals. Is Rex an animal?"
**Example Code 2:**
```python
from z3 import *
s = Solver()
Object = DeclareSort('Object')
Dog = Function('Dog', Object, BoolSort())
Animal = Function('Animal', Object, BoolSort())
rex = Const('rex', Object)
x = Const('x', Object)

# Premise: All dogs are animals
s.add(ForAll(x, Implies(Dog(x), Animal(x))))

# Target: Is Rex an animal?
target = Animal(rex)

s.push()
s.add(Not(target))
if s.check() == unsat:
    RESULT = {"answer": "Yes", "explanation": "The negation of the target leads to a contradiction, so it must be true."}
else:
    s.pop()
    s.push()
    s.add(target)
    if s.check() == unsat:
        RESULT = {"answer": "No", "explanation": "The target leads to a contradiction, so it must be false."}
    else:
        RESULT = {"answer": "Uncertain", "explanation": "Both the target and its negation are satisfiable, so we cannot determine the answer."}
s.pop()
```

**Example Problem 3 (Open-Ended / Variable Extraction):** "Alice is 30 years old. Bob is 25. Who is older?"
**Example Code 3:**
```python
from z3 import *
s = Solver()
Object = DeclareSort('Object')
Age = Function('Age', Object, IntSort())
Alice = Const('Alice', Object)
Bob = Const('Bob', Object)
Person = Const('Person', Object)

s.add(Age(Alice) == 30)
s.add(Age(Bob) == 25)

# We want to find who is older (Age > 25)
s.add(Age(Person) > Age(Bob))

if s.check() == sat:
    m = s.model()
    # Evaluate the person who satisfies the condition
    # For simple string extraction from Z3 constants:
    answer = str(m.evaluate(Person, model_completion=True))
    RESULT = {"answer": answer, "explanation": f"{answer} is older than Bob because their age is {m.evaluate(Age(Person))}."}
else:
    RESULT = {"answer": "Uncertain", "explanation": "Could not determine who is older."}
```

Now solve this problem:

Problem: {{ question }}

Premises:
{% for premise in premises %}
- {{ premise }}
{% endfor %}

**Python Code:**
```python
(your Python code that sets RESULT)
```
""")



LOGIC_SYMBOLIC_TEMPLATE = Template("""
You are an expert logician. Your task is to translate the logic problem into First-Order Logic (FOL).

IMPORTANT RULES:
1. Extraction & Naming
- Extract all predicates (yielding True/False) and functions (yielding numeric/quantifiable values).
- Use strictly `snake_case` for names (e.g., `is_tall(x)`, `score(x)`).
- For functions, explicitly state the return type (e.g., `score(x): Int`, `gpa(x): Real`).
- Preserve relation arity and variable binding. Use multi-argument predicates where the text relates distinct entities, e.g. `lab_access(person, facility)`.
{% if global_glossary %}
- Reuse standardized predicates from the GLOBAL GLOSSARY for concepts it covers. An option may introduce a genuinely new predicate, but it must remain logically independent unless a premise connects it.
**GLOBAL GLOSSARY:**
{% for k, v in global_glossary.items() %}
- `{{ k }}`: {{ v }}
{% endfor %}
{% endif %}

2. Translation & Syntax
- Use standard FOL syntax:
  * Quantifiers: `∀` (all), `∃` (exists)
  * Connectives: `∧` (and), `∨` (or), `→` (implies), `↔` (biconditional), `¬` (not)
- Map numeric comparisons strictly as follows:
  * "more/greater than X" -> `> X`
  * "less/fewer than X" -> `< X`
  * "at least/minimum of X" -> `>= X`
  * "at most/maximum of X" -> `<= X`
  * "exactly/equal/is X" -> `= X`
  * "not equal/other than X" -> `!= X`
- When a premise mentions a specific entity (e.g., "Sophia"), reuse the EXACT general predicate/function name and substitute 'x' with the lowercase entity name (e.g., `is_tall(sophia)`). DO NOT create new predicates like `is_sophia_tall()`.
- Keep constants and predicates in separate namespaces. Entity names, pronouns, and temporal/deictic modifiers are constants or context; never declare them as Boolean predicates.
- For domain-restricted rules (e.g., "If a drone has..."), either omit the domain predicate if it applies universally in context (e.g. `∀x(H(x) → L(x))`), or ensure domain membership is explicitly added as a premise for specific entities (e.g. `is_drone(drone_x)`).
- Preserve semantic roles across an implication. If the condition describes weather but the consequence describes the ground, do not apply both predicates to a temporal constant; represent the affected ground/context explicitly.
- Preserve nested quantifier scope exactly. Universal-existential statements require distinct variables, e.g. `∀x(researcher(x) → ∃y(lab_access(x,y) ∧ secure_facility(y)))`.
- If a premise asserts a standalone fact without an "if-then" condition, emit a grounded fact or a context proposition, never a universal fact and never a tautology.
- A new positive concept in an option is an independent predicate. For example, "snowing" must not become `¬rain`; use negation only when the text explicitly negates the same proposition.
- CRITICAL: DO NOT nest predicates inside other predicates! For example, NEVER write `P(Q(x) ∧ R(x))`. You MUST write them independently joined by logical connectives: `Q(x) ∧ R(x) ∧ P(x)`. All predicates must be evaluated independently at the top level.
- Do NOT hallucinate or add predicates/functions not explicitly mentioned in the text.

{% if options %}
**Options to translate:**
{% for opt in options %}
- {{ opt }}
{% endfor %}
{% endif %}

3. Execution
- Use the pre-extracted Semantics (Intent, Condition, Target) to guide your translation.
- Re-use the existing `premises_fol` exactly if provided. If not, translate `premises` into `premises_fol`.
- Translate the question into `target_fol` (if yes_no).
- Translate the options into `options_fol` (if multiple_choice), strictly following the order of the **Options to translate** provided above. Do NOT hallucinate options that are not in the list.

Return ONLY a valid JSON object matching this schema exactly:

```json
{
  "translation": {
    "predicates": ["predicate_1(x)", "predicate_2(x)"],
    "functions": ["function_name(x): Int"],
    "premises_fol": ["formula1", "formula2"],
    "condition_fol": "",
    "target_fol": "formula",
    "options_fol": []
  }
}
```

Now solve this problem:

Problem: {{ question }}

Pre-extracted Semantics:
- Intent: {{ semantics.intent }}
- Condition: {{ semantics.condition }}
- Target: {{ semantics.target }}
- Query Type: {{ semantics.query_type }}

Premises:
{% for premise in premises %}
- {{ premise }}
{% endfor %}

{% if premises_fol %}
Already Translated Premises FOL:
{% for fol in premises_fol %}
- {{ fol }}
{% endfor %}
(CRITICAL: Do not change these premises. Only translate the target and options using the EXACT SAME predicates and entities as above.)
{% endif %}

**JSON Output:**
""")
