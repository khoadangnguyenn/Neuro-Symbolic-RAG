"""Proknow-RAG Jinja2 Templates."""

from jinja2 import Template

PHYSICS_TEMPLATE = Template("""
You are an expert physicist and Python programmer. Solve the physics problem using the sympy library.

IMPORTANT RULES:
- Use sympy.symbols() for unknowns.
- Use sympy.Eq() for equations and sympy.solve() to solve.
- CRITICAL RULE (Coordinate Guardrail): For ANY spatial, 2D, 3D, or geometric problem (e.g. points, distances, electric/magnetic forces, vectors), you MUST assign explicit (x, y) or (x, y, z) coordinates to ALL points. 
- NEVER attempt to guess distances or add/subtract vector magnitudes directly. You must compute distances using the distance formula: `sqrt((x2-x1)**2 + (y2-y1)**2)`.
- Always decompose vectors into their X and Y components to find the net vector before taking the magnitude.
- You MUST set a global variable RESULT = {"answer": "...", "explanation": "..."}.
- Do NOT use print(). Only set RESULT.
- The "answer" value must be a number with its unit (e.g. "2.5 J", "100 V").
- The Retrieved Formulas/Rules are already pre-compiled or structured. DO NOT re-translate them. Just use them directly to translate the User Question into the final solver code.

Here is a complete working example of the Coordinate Guardrail:

**Example Problem:** "Two point charges q1=5C at origin and q2=-5C at x=3, y=0. Find force on q3=1C at x=0, y=4."
**Example Code:**
```python
import sympy
k = 9e9
q1, q2, q3 = 5, -5, 1
# Assign Coordinates
p1 = (0, 0)
p2 = (3, 0)
p3 = (0, 4)
# Distances
r13 = sympy.sqrt((p3[0]-p1[0])**2 + (p3[1]-p1[1])**2)
r23 = sympy.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2)
# Magnitudes
F13 = k * abs(q1*q3) / r13**2
F23 = k * abs(q2*q3) / r23**2
# Vectors (Target - Source)
u13 = ((p3[0]-p1[0])/r13, (p3[1]-p1[1])/r13)
u23 = ((p3[0]-p2[0])/r23, (p3[1]-p2[1])/r23)
# Force components
Fx = F13 * u13[0] - F23 * u23[0] # Note: attraction/repulsion signs
Fy = F13 * u13[1] - F23 * u23[1]
net_F = sympy.sqrt(Fx**2 + Fy**2)
RESULT = {"answer": f"{net_F.evalf()} N", "explanation": "Calculated net force using vector addition."}
```

Now solve this problem:

Problem: {{ question }}

Retrieved Formulas / Rules:
{% for premise in premises %}
- {{ premise }}
{% endfor %}

**Reasoning:**
(your step-by-step reasoning)

**Python Code:**
```python
(your sympy code that sets RESULT)
```
""")

LOGIC_TEMPLATE = Template("""
You are an expert logician and Python programmer. Solve the logic problem using the z3-solver library.

IMPORTANT RULES:
- You must declare ALL variables, functions, and sorts before using them! 
- Use `Bool('name')` for propositions.
- For First-Order Logic (predicates like P(x)), declare a sort and the function:
  `Object = DeclareSort('Object')`
  `P = Function('P', Object, BoolSort())`
  `x = Const('x', Object)`
- Add ALL premises as constraints with s.add().
- For implication, ALWAYS use Implies(a, b). NEVER use the >> operator.
- Check satisfiability with s.check(). If sat, read the model with s.model().
- You MUST set a global variable RESULT = {"answer": "...", "explanation": "..."}.
- Do NOT use print(). Only set RESULT.
- The retrieved Premises/Subgraph are already pre-compiled Z3 rules or extracted facts. DO NOT re-translate them into FOL step-by-step. Just incorporate them and translate the final User Question into an `s.add(...)` condition.

Here is a complete working example:

**Example Problem:** "All dogs are animals. Rex is a dog. Is Rex an animal?"
**Example Code:**
```python
from z3 import *
s = Solver()
rex_is_dog = Bool('rex_is_dog')
rex_is_animal = Bool('rex_is_animal')
# Premise: All dogs are animals (if dog then animal)
s.add(Implies(rex_is_dog, rex_is_animal))
# Premise: Rex is a dog
s.add(rex_is_dog == True)
# Query: Is Rex an animal?
s.add(rex_is_animal == True)
if s.check() == sat:
    RESULT = {"answer": "Yes", "explanation": "Rex is a dog, all dogs are animals, so Rex is an animal."}
else:
    RESULT = {"answer": "No", "explanation": "The premises lead to a contradiction."}
```

Now solve this problem:

Problem: {{ question }}

Premises:
{% for premise in premises %}
- {{ premise }}
{% endfor %}

**Reasoning:**
(your step-by-step reasoning)

**Python Code:**
```python
(your z3 code that sets RESULT)
```
""")
