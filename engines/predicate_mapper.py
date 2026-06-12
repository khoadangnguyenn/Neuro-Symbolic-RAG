"""Rule-based predicate extraction and NL→FOL symbol mapping.

This module provides deterministic vocabulary extraction from premises and
question options, eliminating the need for LLM to invent predicate names.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Extract FOL symbols from question Options (A. ∀x (G(x) → ...) )
# ---------------------------------------------------------------------------

def extract_fol_symbols_from_options(question: str) -> List[str]:
    """Extract single-letter FOL predicate symbols from Options in the question.
    
    Handles patterns like:
      A. ∀x (G(x) → (S(x) ∧ C(x)))
      B. ∃x (¬O(x) ∧ R(x))
    
    Returns: ["G", "S", "C", "O", "R", "B"] (deduplicated, sorted)
    """
    # Match uppercase single-letter followed by '(' — standard FOL predicate
    symbols = set(re.findall(r'\b([A-Z])\s*\(', question))
    # Remove common quantifier variables (x, y, z are NOT predicates)
    symbols -= {'x', 'y', 'z', 'X', 'Y', 'Z'}
    return sorted(symbols)


def extract_snake_predicates_from_options(question: str) -> List[str]:
    """Extract snake_case predicates from Options (e.g., has_camera(x)).
    
    Returns: ["has_camera", "has_gps"] (deduplicated, sorted)
    """
    preds = set(re.findall(r'\b([a-z][a-z_0-9]*)\s*\(', question))
    # Remove common non-predicates
    preds -= {'x', 'y', 'z', 'not', 'and', 'or', 'implies', 'iff'}
    return sorted(preds)


def extract_options_fol(question: str) -> List[str]:
    """Extract FOL formulas from A., B., C., D. lines in the question.
    
    Returns: ["∀x (G(x) → (S(x) ∧ C(x)))", "∃x (¬O(x) ∧ R(x))", ...]
    """
    options = []
    # Match "A. <formula>" or "A) <formula>" patterns
    matches = re.findall(r'(?:^|\n)\s*([A-Z])[.)\s]+(.+?)(?=\n\s*[A-Z][.)\s]|\Z)', question, re.DOTALL)
    for letter, formula in matches:
        clean = formula.strip()
        if clean:
            options.append(clean)
    return options


# ---------------------------------------------------------------------------
# 2. Extract NL property keywords from premises  
# ---------------------------------------------------------------------------

def _normalize_property(prop: str) -> str:
    """Convert a NL property phrase to snake_case identifier."""
    prop = prop.strip().lower()
    prop = re.sub(r'\b(a|an|the)\b', '', prop).strip()
    prop = re.sub(r'[^a-z0-9\s]', '', prop)
    prop = re.sub(r'\s+', '_', prop).strip('_')
    return prop


def extract_nl_properties(premises_nl: List[str]) -> List[str]:
    """Extract unique NL property phrases from premises.
    
    Scans for patterns like:
      - "has X" / "have X" / "has a X"
      - "lacks X" / "does not have X"  
      - "is X" / "are X"
    
    Returns: ["image_stabilization", "high_quality_camera", "long_remote_control_range", ...]
    """
    properties: Set[str] = set()
    
    for premise in premises_nl:
        premise_lower = premise.lower().strip().rstrip('.')
        
        # Pattern 1: "has/have (a/an) <property>"
        for match in re.finditer(r'(?:has|have)\s+(?:a\s+|an\s+)?(.+?)(?:\s*[,.]|\s+then\b|\s+and\b|\s+or\b|$)', premise_lower):
            prop = _normalize_property(match.group(1))
            if prop and len(prop) > 1:
                properties.add(prop)
        
        # Pattern 2: "lacks <property>" / "does not have <property>"
        for match in re.finditer(r'(?:lacks|does\s+not\s+have)\s+(?:a\s+|an\s+)?(.+?)(?:\s*[,.]|\s+then\b|\s+and\b|\s+or\b|$)', premise_lower):
            prop = _normalize_property(match.group(1))
            if prop and len(prop) > 1:
                properties.add(prop)
    
    return sorted(properties)


# ---------------------------------------------------------------------------
# 3. Build NL↔FOL symbol mapping
# ---------------------------------------------------------------------------

def build_predicate_map(
    nl_properties: List[str],
    fol_symbols: List[str],
    premises_nl: List[str],
    question: str,
) -> Dict[str, str]:
    """Attempt to map NL property names to single-letter FOL symbols.
    
    Uses heuristic: match first letter of the most distinctive word.
    E.g.:
      G → gps_navigation
      S → image_stabilization (S for Stabilization)
      C → high_quality_camera (C for Camera)
      O → obstacle_avoidance (O for Obstacle)
      R → long_remote_control_range (R for Remote/Range)
      B → long_battery_life (B for Battery)
    
    Returns: {"G": "gps_navigation", "S": "image_stabilization", ...}
    """
    if not fol_symbols or not nl_properties:
        return {}
    
    # Build keyword → property index
    # Strategy: for each FOL symbol letter, find the NL property whose
    # "most distinctive word" starts with that letter
    mapping: Dict[str, str] = {}
    used_props: Set[str] = set()
    
    for sym in fol_symbols:
        letter = sym.upper()
        best_match = None
        best_score = 0
        
        for prop in nl_properties:
            if prop in used_props:
                continue
            words = prop.split('_')
            for word in words:
                if word and word[0].upper() == letter:
                    # Score by word length (longer = more distinctive)
                    score = len(word)
                    if score > best_score:
                        best_score = score
                        best_match = prop
        
        if best_match:
            mapping[sym] = best_match
            used_props.add(best_match)
    
    return mapping


# ---------------------------------------------------------------------------
# 4. Public API: one-call extraction 
# ---------------------------------------------------------------------------

def extract_predicates_from_context(
    question: str,
    premises_nl: List[str],
) -> Dict:
    """Extract all predicate information needed for FOL translation.
    
    Returns dict with:
      - fol_symbols: ["G", "S", "C", ...] from Options
      - snake_predicates: ["has_camera", ...] from Options (if snake_case)
      - nl_properties: ["image_stabilization", ...] from premises
      - predicate_map: {"G": "gps_navigation", ...}
      - options_fol: ["∀x (G(x) → ...)", ...] from Options
      - allowed_predicates: combined list for LLM prompt locking
    """
    fol_symbols = extract_fol_symbols_from_options(question)
    snake_predicates = extract_snake_predicates_from_options(question)
    nl_properties = extract_nl_properties(premises_nl)
    options_fol = extract_options_fol(question)
    
    predicate_map = build_predicate_map(nl_properties, fol_symbols, premises_nl, question)
    
    # Build the final "allowed predicates" list for LLM vocabulary locking
    if fol_symbols:
        allowed = fol_symbols  # Use single-letter symbols
    elif snake_predicates:
        allowed = snake_predicates
    else:
        # Fallback: use NL properties as predicate names
        allowed = nl_properties
    
    return {
        "fol_symbols": fol_symbols,
        "snake_predicates": snake_predicates,
        "nl_properties": nl_properties,
        "predicate_map": predicate_map,
        "options_fol": options_fol,
        "allowed_predicates": allowed,
    }
