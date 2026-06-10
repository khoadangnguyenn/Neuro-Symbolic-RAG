"""Dynamic Rules Feedback Loop."""

from __future__ import annotations

import re
import uuid

from exact_pipeline.knowledge.graph_db import HybridDB
from exact_pipeline.knowledge.knowledge import FormulaCard


def extract_and_write_back_physics(code: str, hybrid_db: HybridDB) -> None:
    """Extract derived formulas from successful code and add to graph."""
    # Heuristic to find sympy Eq statements like Eq(v, F / m * t)
    eq_matches = re.findall(r"(?:sympy\.)?Eq\(([^,]+),\s*(.+?)\)", code)
    if not eq_matches:
        return

    for lhs, rhs in eq_matches:
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        # Don't add simple variable assignments
        if rhs.replace(".", "").isdigit():
            continue
            
        formula_id = f"dynamic_physics_{uuid.uuid4().hex[:8]}"
        trigger_terms = tuple(set([lhs] + re.findall(r"\b[A-Za-z_]+\b", rhs)))
        
        card = FormulaCard(
            formula_id=formula_id,
            family="dynamic",
            expression=f"{lhs} = {rhs}",
            premise="Dynamically derived formula from verified execution.",
            trigger_terms=trigger_terms,
        )
        hybrid_db.add_rule(formula_id, card)


def extract_and_write_back_logic(code: str, hybrid_db: HybridDB) -> None:
    """Extract logic constraints and add to graph."""
    # Heuristic to find Z3 constraints added to the solver
    rules = re.findall(r"(?:s|solver)\.add\((.+?)\)", code)
    if not rules:
        return

    for rule in rules:
        # Only add complex structural rules, not simple ground facts
        if "Implies" in rule or "ForAll" in rule or "Or" in rule or "Not(" in rule:
            rule_id = f"dynamic_logic_{uuid.uuid4().hex[:8]}"
            hybrid_db.add_rule(rule_id, rule)
