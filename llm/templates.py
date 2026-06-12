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
- IMPORTANT: Use `float()` instead of `.evalf()` for the final answer to avoid AttributeError when standard Python floats are generated.
- Do NOT use print(). Only set RESULT.
- The "answer" value must be a number with its unit (e.g. "2.5 J", "100 V").
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
RESULT = {"answer": f"{float(net_F)} N", "explanation": "Calculated net force using vector addition."}
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
- You MUST declare ALL variables, functions, and sorts before using them! Use `Bool('name')` for propositions.
- You MUST declare ALL variables (e.g. `x = Const('x', Object)`) before using them!
- For ALL predicates used in the problem AND the multiple-choice options (e.g. `C(x)`, `B(x)`), you MUST declare them using `Function('name', Object, BoolSort())`. Do NOT forget predicates that only appear in the options!
- For First-Order Logic, declare a sort and the function: `Object = DeclareSort('Object')`, `P = Function('P', Object, BoolSort())`, `x = Const('x', Object)`.
- When translating universal statements ('If a drone...', 'Every drone...'), you MUST strictly wrap the implications in ForAll(x, ...). For example: `ForAll(x, Implies(Not(G(x)), Not(H(x))))`.
- For logical negation, ALWAYS use `Not(...)`. NEVER use the bitwise `~` operator (e.g. do not use `~G(x)`).
- Add ALL premises as constraints with s.add(). For implication, ALWAYS use Implies(a, b). NEVER use the >> operator.
- If the user provides multiple-choice options (A, B, C, D), DO NOT just check if the premises are SAT. You MUST iterate through each option to prove Entailment. To prove an option is correct, you must push the NEGATION of the option into the solver (`s.add(Not(option))`). If the result is `unsat`, that option is the correct logical conclusion. Use `s.push()` and `s.pop()` for each option.
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

LOGIC_NETWORKX_TEMPLATE = Template("""
You are an expert logician and Python programmer. Solve the logic problem using the networkx library.

IMPORTANT RULES:
- DO NOT use z3! Use `networkx` to build a directed graph (nx.DiGraph) of the implications.
- Nodes are strings (facts/conditions). Use `G.add_edge(A, B)` to represent `A -> B`.
- Use `nx.shortest_path_length(G, source, target)` to measure path length.
- "Strongest conclusion" = MAXIMUM path length from the initial facts.
- "Fewest premises" = MINIMUM path length from the initial facts.
- You MUST set a global variable RESULT = {"answer": "...", "explanation": "...", "premises_used": [indices_of_used_premises]}
- Note: premises_used should be a list of the 0-based indices corresponding to the premises that form the path you found.
- Do NOT use print(). Only set RESULT.
- Output ONLY valid Python/NetworkX code enclosed in python blocks. NO explanations, NO pleasantries, NO markdown outside the code block.

Here is a complete working example for NetworkX:

**Example Problem:** "If A then B. If B then C. Which is the strongest conclusion from A?"
**Example Code:**
```python
import networkx as nx
G = nx.DiGraph()
G.add_edge("A", "B", premise_index=0)
G.add_edge("B", "C", premise_index=1)
# Find longest path from A
paths = nx.single_source_shortest_path_length(G, "A")
# The strongest conclusion is the furthest node
strongest_node = max(paths, key=paths.get)
shortest_path = nx.shortest_path(G, "A", strongest_node)
used_indices = []
for i in range(len(shortest_path)-1):
    u = shortest_path[i]
    v = shortest_path[i+1]
    if "premise_index" in G[u][v]:
        used_indices.append(G[u][v]["premise_index"])

RESULT = {"answer": strongest_node, "explanation": f"The longest chain of inference from A leads to {strongest_node}.", "premises_used": used_indices}
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
- Use the pre-extracted Semantics (Intent, Condition, Target) to guide your translation.
- Extract all predicates used in the premises and options. Write them in `snake_case(x)`.
- Re-use the existing `premises_fol` if provided. If not, translate `premises` into `premises_fol`.
- Translate the question into `target_fol` (if yes_no).
- Translate the options into `options_fol` (if multiple_choice), strictly following the order A, B, C, D.
- Return ONLY a valid JSON object matching this schema exactly:

```json
{
  "translation": {
    "predicates": ["predicate_1(x)", "predicate_2(x)"],
    "functions": [],
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
{% for f in premises_fol %}
- {{ f }}
{% endfor %}
{% endif %}

**JSON Output:**
""")
