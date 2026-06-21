"""Deterministic Horn-style reasoning for Type 1 logic questions.

This module is intentionally conservative. It extracts simple unary facts,
Horn rules, negative facts, and numeric facts from the supplied premises, then
uses forward chaining before any generative fallback is trusted. The goal is
not to replace Z3 for full FOL, but to make common educational logic problems
solver-owned instead of LLM-owned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import re

from exact_pipeline.core.models import PipelineResult


CONTEXT_ENTITY = "__context__"


@dataclass(frozen=True)
class Atom:
    name: str
    entity: str = CONTEXT_ENTITY
    negated: bool = False

    def positive(self) -> "Atom":
        return Atom(self.name, self.entity, False)

    def negate(self) -> "Atom":
        return Atom(self.name, self.entity, not self.negated)


@dataclass(frozen=True)
class Rule:
    antecedents: Tuple[Atom, ...]
    consequent: Atom
    premise_index: int


@dataclass(frozen=True)
class NumericFact:
    subject_key: str
    subject_label: str
    measure_key: str
    value: str
    premise_index: int


@dataclass(frozen=True)
class Disjunction:
    """A finite clause whose alternatives share the current entity binding."""

    alternatives: Tuple[Atom, ...]
    premise_index: int


@dataclass(frozen=True)
class ExistentialRequirement:
    """``kind(x) -> exists y: relation/object_type(x, y)`` projection."""

    antecedent: Atom
    object_type: str
    premise_index: int


@dataclass(frozen=True)
class ExistentialAbsence:
    """A grounded assertion that an entity has no object of a given type."""

    entity: str
    object_type: str
    premise_index: int


@dataclass
class ReasoningState:
    facts: Dict[Atom, Tuple[int, ...]] = field(default_factory=dict)
    rules: List[Rule] = field(default_factory=list)
    disjunctions: List[Disjunction] = field(default_factory=list)
    existential_requirements: List[ExistentialRequirement] = field(default_factory=list)
    existential_absences: List[ExistentialAbsence] = field(default_factory=list)
    inconsistency_support: Tuple[int, ...] = ()
    numeric_facts: List[NumericFact] = field(default_factory=list)
    uncertainty_notes: Dict[Atom, int] = field(default_factory=dict)
    entity_labels: Dict[str, str] = field(default_factory=dict)
    generated_fol: List[str] = field(default_factory=list)
    domain_terms: set[str] = field(default_factory=set)
    predicate_aliases: Dict[str, str] = field(default_factory=dict)
    pending_universals: List[Tuple[str, int]] = field(default_factory=list)


@dataclass(frozen=True)
class HornCapability:
    supported: bool
    reasons: Tuple[str, ...] = ()


def assess_horn_capability(
    *,
    question: str,
    premises_nl: Sequence[str],
    premises_fol: Sequence[str],
    options: Sequence[str] = (),
) -> HornCapability:
    """Decide whether the input belongs to the sound deterministic fragment.

    The local engine intentionally owns only unary Horn rules, grounded unary
    facts, explicit negation, and the small compositional query operators it
    implements.  Rich FOL is routed to the symbolic solver instead of being
    partially parsed and mistaken for a complete theory.
    """

    reasons: List[str] = []
    for raw in premises_fol or ():
        fol = _normalize_fol(str(raw))
        if not fol or fol.startswith("/*") or "[NEEDS TRANSLATION]" in fol:
            continue
        if re.search(r"(?:∃|\bexists\b)", fol, flags=re.I):
            reasons.append("existential_quantifier")
        quantifiers = re.findall(r"(?:∀|∃|\bforall\b|\bexists\b)", fol, flags=re.I)
        if len(quantifiers) > 1:
            reasons.append("nested_or_multiple_quantifiers")
        if re.search(r"(?:↔|<->|∨|(?<!\|)\|(?!\|))", fol):
            reasons.append("non_horn_connective")
        for arguments in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(([^()]*)\)", fol):
            if "," in arguments:
                reasons.append("multi_argument_predicate")
                break

    combined_nl = "\n".join([*(str(p) for p in premises_nl), str(question), *(str(o) for o in options)])
    # Natural-language surface markers are not capability failures by
    # themselves.  The controlled parser below owns iff, disjunction,
    # whenever/unless, conjunction and common existential-cardinality forms.
    # Completeness is checked after parsing; an unrepresented sentence causes
    # a safe fallback to typed AST + Z3 instead of a partial Horn answer.
    if re.search(
        r"\b(?:has|have|access(?:es)?|related\s+to)\b.+\bany\s+[A-Za-z]",
        str(question),
        flags=re.I,
    ):
        reasons.append("existential_query")

    return HornCapability(not reasons, tuple(sorted(set(reasons))))


def try_deterministic_logic(
    *,
    question: str,
    premises_nl: Sequence[str],
    premises_fol: Sequence[str],
    options: Sequence[str],
    query_type: str,
    intent: str,
    learned_aliases: Optional[Mapping[str, str]] = None,
) -> Optional[PipelineResult]:
    """Return a solver-owned answer when the deterministic layer can decide it."""

    capability = assess_horn_capability(
        question=question,
        premises_nl=premises_nl,
        premises_fol=premises_fol,
        options=options,
    )
    if not capability.supported:
        return None

    state = build_reasoning_state(premises_nl, premises_fol, learned_aliases=learned_aliases)
    if not state.facts and not state.rules and not state.numeric_facts:
        return None

    represented = {rule.premise_index for rule in state.rules if rule.premise_index >= 0}
    represented.update(
        index
        for support in state.facts.values()
        for index in support
        if index >= 0
    )
    represented.update(fact.premise_index for fact in state.numeric_facts)
    represented.update(item.premise_index for item in state.disjunctions)
    represented.update(item.premise_index for item in state.existential_requirements)
    represented.update(item.premise_index for item in state.existential_absences)
    represented.update(state.uncertainty_notes.values())
    rule_like = re.compile(
        r"^(?:if|all|every|any|each|no)\b|"
        r"\b(?:then|implies|whenever|unless|either|if\s+and\s+only\s+if)\b",
        flags=re.I,
    )
    if any(
        rule_like.search(_clean_sentence(premise)) and index not in represented
        for index, premise in enumerate(premises_nl)
    ):
        return None

    _close_under_rules(state)

    if any(atom.negate() in state.facts for atom in state.facts):
        return None

    if query_type == "multiple_choice":
        result = _answer_multiple_choice(question, options, premises_nl, state, intent)
    elif query_type == "yes_no_uncertain":
        result = _answer_yes_no(question, premises_nl, state)
    else:
        result = _answer_open_ended(question, premises_nl, state)

    if result is not None:
        result.metadata["executor"] = "deterministic-horn"
        induced_aliases = {
            surface: canonical
            for surface, canonical in state.predicate_aliases.items()
            if surface != canonical
        }
        result.metadata["induced_schema"] = induced_aliases
        if result.reasoning is None:
            derived = [
                {
                    "predicate": atom.name,
                    "entity": state.entity_labels.get(atom.entity, atom.entity),
                    "negated": atom.negated,
                    "premises_used": list(support),
                }
                for atom, support in state.facts.items()
                if len(support) > 1 and set(support).issubset(set(result.premises_used))
            ]
            result.reasoning = {
                "type": "horn_forward_chaining",
                "derived_facts": derived,
                "induced_schema": induced_aliases,
            }
    return result


def build_reasoning_state(
    premises_nl: Sequence[str],
    premises_fol: Sequence[str] = (),
    *,
    learned_aliases: Optional[Mapping[str, str]] = None,
) -> ReasoningState:
    state = ReasoningState()

    for idx, premise in enumerate(premises_nl):
        _ingest_nl_premise(str(premise), idx, state)

    _ingest_deferred_universals(state)

    for idx, fol in enumerate(premises_fol or []):
        text = str(fol).strip()
        if not text or text.startswith("/*") or "[NEEDS TRANSLATION]" in text:
            continue
        if "[PRE-TRANSLATED FOL]" in text:
            text = text.replace("[PRE-TRANSLATED FOL]", "").strip()
        if _ingest_fol_formula(text, idx, state):
            state.generated_fol.append(text)

    _induce_predicate_schema(state, learned_aliases or {})
    _induce_directional_projections(state)
    return state


def _ingest_nl_premise(premise: str, idx: int, state: ReasoningState) -> None:
    text = _clean_sentence(premise)
    if not text:
        return

    uncertainty_atom = _parse_uncertainty_note(text, state.domain_terms)
    if uncertainty_atom:
        state.uncertainty_notes[uncertainty_atom] = idx
        return

    numeric = _parse_numeric_fact(text, idx)
    if numeric:
        state.numeric_facts.append(numeric)
        _remember_entity(state, numeric.subject_key, numeric.subject_label)
        return

    existential_requirement = _parse_existential_requirement(text, idx)
    if existential_requirement:
        state.existential_requirements.append(existential_requirement)
        return

    existential_absence = _parse_existential_absence(text, idx)
    if existential_absence:
        state.existential_absences.append(existential_absence)
        subject, _ = _extract_leading_entity(text, state.domain_terms)
        if subject:
            _remember_entity(state, existential_absence.entity, subject)
        return

    disjunction = _parse_disjunction(text, idx)
    if disjunction:
        state.disjunctions.append(disjunction)
        return

    biconditional_rules = _parse_biconditional_rules(text, idx)
    if biconditional_rules:
        state.rules.extend(biconditional_rules)
        return

    whenever_rule = _parse_whenever_rule(text, idx)
    if whenever_rule:
        state.rules.append(whenever_rule)
        return

    unless_rule = _parse_unless_rule(text, idx)
    if unless_rule:
        state.rules.append(unless_rule)
        return

    rule = _parse_if_then_rule(text, idx)
    if rule:
        state.rules.append(rule)
        projected = _parse_requirement_projection(text, idx, rule)
        if projected:
            state.rules.append(projected)
        state.domain_terms.update(_infer_rule_domain(text))
        return

    rules = _parse_who_rule(text, idx)
    if rules:
        state.rules.extend(rules)
        return

    rule = _parse_participle_universal_rule(text, idx)
    if rule:
        state.rules.append(rule)
        return

    rule = _parse_universal_class_rule(text, idx)
    if rule:
        state.rules.append(rule)
        return

    rule = _parse_modal_universal_rule(text, idx)
    if rule:
        state.rules.append(rule)
        return

    # A sentence can start with a surface quantifier and still be a grounded
    # fact when it names a concrete entity (for example, "All affected Atlas
    # passwords have been reset").  Only defer genuinely class-level clauses.
    grounded_subject, _ = _extract_leading_entity(text, state.domain_terms)
    if re.match(r"^(?:all|every|any)\b", text, flags=re.I) and not grounded_subject:
        state.pending_universals.append((text, idx))
        return

    atoms = _parse_grounded_conjunction(text, state.domain_terms)
    if not atoms:
        atom = _statement_to_atom(text, state.domain_terms)
        atoms = [atom] if atom else []
    for atom in atoms:
        subject, _ = _extract_leading_entity(text)
        if subject:
            _remember_entity(state, atom.entity, subject)
        _add_fact(state, atom, [idx])


def _parse_uncertainty_note(text: str, domain_terms: Iterable[str] = ()) -> Optional[Atom]:
    match = re.search(
        r"\bno\s+premise\s+states\s+whether\s+(.+)$", text, flags=re.I
    )
    if not match:
        return None
    return _statement_to_atom(match.group(1).strip(), domain_terms)


def _parse_numeric_fact(text: str, idx: int) -> Optional[NumericFact]:
    if re.match(r"^(?:if|whenever|unless|all|every|any|each|no)\b", text, flags=re.I):
        return None
    patterns = [
        r"^(?P<subject>.+?)\s+(?:has|have|contains|include|includes)\s+"
        r"(?P<value>[-+]?\d+(?:\.\d+)?)\s+(?P<measure>.+)$",
        r"^(?P<subject>.+?)\s+(?:is|are)\s+"
        r"(?P<value>[-+]?\d+(?:\.\d+)?)\s+(?P<measure>.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.I)
        if not match:
            continue
        subject_label = _strip_articles(match.group("subject")).strip()
        if not subject_label:
            continue
        measure = _normalize_measure(match.group("measure"))
        if not measure:
            continue
        return NumericFact(
            subject_key=_entity_key(subject_label),
            subject_label=subject_label,
            measure_key=measure,
            value=match.group("value"),
            premise_index=idx,
        )
    return None


def _parse_disjunction(text: str, idx: int) -> Optional[Disjunction]:
    """Parse ordinary finite alternatives without assuming exclusivity."""

    shared = re.match(
        r"^(.+?)\s+(?:is|are)\s+either\s+(.+?)\s+or\s+(.+)$",
        text,
        flags=re.I,
    )
    if shared:
        subject, left, right = shared.groups()
        domain = _strip_quantifier(_strip_articles(subject)).strip()
        left_atom = _clause_to_atom(left, domain=domain)
        right = re.sub(r"^(?:it|they|that|the\s+same)\s+", "", right, flags=re.I)
        right_atom = _clause_to_atom(right, domain=domain)
        if left_atom and right_atom:
            return Disjunction((left_atom, right_atom), idx)

    leading = re.match(r"^either\s+(.+?)\s+or\s+(.+)$", text, flags=re.I)
    if leading:
        alternatives = tuple(
            atom
            for atom in (
                _statement_to_atom(leading.group(1)),
                _statement_to_atom(leading.group(2)),
            )
            if atom
        )
        if len(alternatives) == 2:
            return Disjunction(alternatives, idx)
    return None


def _parse_biconditional_rules(text: str, idx: int) -> List[Rule]:
    match = re.match(r"^(.+?)\s+(?:if\s+and\s+only\s+if|iff)\s+(.+)$", text, flags=re.I)
    if not match:
        return []
    left_text, right_text = match.groups()
    domain_match = re.match(r"^(?:a|an|the|any|every|all)\s+(.+)$", left_text, flags=re.I)
    domain = ""
    if domain_match:
        words = domain_match.group(1).split()
        auxiliaries = {"is", "are", "has", "have", "can", "may", "must", "should", "will"}
        for position, word in enumerate(words):
            if word.lower() in auxiliaries or _looks_like_predicate_head(word):
                domain = " ".join(words[:position])
                left_text = " ".join(words[position:])
                break
    left = _clause_to_atom(left_text, domain=domain)
    right = _clause_to_atom(right_text, domain=domain)
    if not left or not right:
        return []
    return [Rule((left,), right, idx), Rule((right,), left, idx)]


def _parse_whenever_rule(text: str, idx: int) -> Optional[Rule]:
    """``Q whenever P`` has the implication direction ``P -> Q``."""

    leading = re.match(r"^whenever\s+(.+?),\s*(.+)$", text, flags=re.I)
    if leading:
        condition_text, conclusion_text = leading.groups()
        domain = _infer_domain(condition_text, conclusion_text)
        condition = _clause_to_atom(condition_text, domain=domain)
        conclusion = _clause_to_atom(conclusion_text, domain=domain)
        if condition and conclusion:
            return Rule((condition,), conclusion, idx)

    match = re.match(r"^(.+?)\s+whenever\s+(.+)$", text, flags=re.I)
    if not match:
        return None
    conclusion_text, condition_text = match.groups()
    condition_text = re.sub(r"^(?:it|they|that|the\s+same)\s+", "", condition_text, flags=re.I)
    conclusion = _clause_to_atom(conclusion_text)
    condition = _clause_to_atom(condition_text)
    if not conclusion or not condition:
        return None
    return Rule((condition,), conclusion, idx)


def _parse_unless_rule(text: str, idx: int) -> Optional[Rule]:
    """Parse necessary-condition forms: ``No P unless Q`` means ``P -> Q``."""

    match = re.match(r"^no\s+(.+?)\s+unless\s+(.+)$", text, flags=re.I)
    if not match:
        return None
    guarded_text, requirement_text = match.groups()
    domain, guarded = _split_rule_tail(guarded_text)
    if not domain or not guarded:
        return None
    requirement_text = re.sub(
        r"^(?:it|they|that|the\s+same)\s+", "", requirement_text, flags=re.I
    )
    guarded_atom = _clause_to_atom(guarded, domain=domain)
    required_atom = _clause_to_atom(requirement_text, domain=domain)
    if not guarded_atom or not required_atom:
        return None
    # The initial "No" scopes over the exception construction, not over P.
    return Rule((guarded_atom.positive(),), required_atom.positive(), idx)


def _parse_modal_universal_rule(text: str, idx: int) -> Optional[Rule]:
    match = re.match(
        r"^(?:all|every|any|each)\s+(.+?)\s+"
        r"(must|can|may|should|will)\s+(.+)$",
        text,
        flags=re.I,
    )
    if not match:
        return None
    domain_text, modal, conclusion_text = match.groups()
    antecedent = _class_phrase_to_atom(domain_text)
    consequent = _clause_to_atom(f"{modal} {conclusion_text}", domain=domain_text)
    if not antecedent or not consequent:
        return None
    return Rule((antecedent,), consequent, idx)


def _parse_existential_requirement(
    text: str, idx: int
) -> Optional[ExistentialRequirement]:
    match = re.match(
        r"^(?:all|every|any)\s+(.+?)\s+(?:has|have|contains?|includes?)\s+"
        r"at\s+least\s+(?:one|1)\s+(.+)$",
        text,
        flags=re.I,
    )
    if not match:
        return None
    antecedent = _class_phrase_to_atom(match.group(1))
    object_type, negated = _phrase_to_predicate(match.group(2))
    if not antecedent or not object_type or negated:
        return None
    return ExistentialRequirement(antecedent, object_type, idx)


def _parse_existential_absence(text: str, idx: int) -> Optional[ExistentialAbsence]:
    match = re.match(r"^(.+?)\s+(?:has|have|contains?)\s+no\s+(.+)$", text, flags=re.I)
    if not match:
        return None
    subject = _strip_articles(match.group(1)).strip()
    if not subject or not re.search(r"[A-Z]", subject):
        return None
    object_type, _ = _phrase_to_predicate(match.group(2))
    if not object_type:
        return None
    return ExistentialAbsence(_entity_key(subject), object_type, idx)


def _parse_grounded_conjunction(
    text: str, domain_terms: Iterable[str] = ()
) -> List[Atom]:
    subject, remainder = _extract_leading_entity(text, domain_terms)
    if not subject or not re.search(r"\s+(?:and|but)\s+", remainder, flags=re.I):
        return []
    pieces = [
        piece.strip(" ,")
        for piece in re.split(r"\s+(?:and|but)\s+", remainder, flags=re.I)
        if piece.strip(" ,")
    ]
    if len(pieces) < 2:
        return []
    entity = _entity_key(subject)
    atoms: List[Atom] = []
    for piece in pieces:
        name, negated = _phrase_to_predicate(piece)
        if not name:
            return []
        atoms.append(Atom(name, entity, negated))
    return atoms


def _parse_if_then_rule(text: str, idx: int) -> Optional[Rule]:
    match = re.match(r"^if\s+(.+?)(?:,\s*)?then\s+(.+)$", text, flags=re.I)
    if not match:
        match = re.match(r"^if\s+(.+?),\s*(.+)$", text, flags=re.I)
    if not match:
        return None

    raw_conditions = match.group(1).strip()
    raw_conclusion = match.group(2).strip()
    domain = _infer_domain(raw_conditions, raw_conclusion)
    clauses = _expand_elliptical_conjuncts(raw_conditions, domain)
    conditions = [
        _clause_to_atom(clause, domain=domain)
        for clause in clauses
    ]
    conclusion = _clause_to_atom(raw_conclusion, domain=domain)
    conditions = [atom for atom in conditions if atom]
    if not conditions or not conclusion:
        return None
    return Rule(tuple(conditions), conclusion, idx)


def _parse_requirement_projection(text: str, idx: int, base_rule: Rule) -> Optional[Rule]:
    """Project an explicit ``not ... without X`` constraint into ``requires X``.

    This is a grammar-level implication, not domain vocabulary: saying an
    action must not occur without a review/permit/check entails that the named
    guard is required for that action.
    """

    match = re.match(r"^if\s+.+?(?:,\s*)?then\s+(.+)$", text, flags=re.I)
    if not match:
        return None
    conclusion = match.group(1).strip()
    guard = re.search(r"\bnot\b.+?\bwithout\s+(.+)$", conclusion, flags=re.I)
    if not guard:
        return None
    name, negated = _phrase_to_predicate(f"requires {guard.group(1)}")
    if not name:
        return None
    return Rule(base_rule.antecedents, Atom(name, CONTEXT_ENTITY, negated), idx)


def _parse_universal_class_rule(text: str, idx: int) -> Optional[Rule]:
    """Parse class-inclusion statements such as "All cats are mammals"."""

    match = re.match(
        r"^(?:all|every|any)\s+(.+?)\s+(?:are|is)\s+(.+)$",
        text,
        flags=re.I,
    )
    if not match:
        return None

    left = _class_phrase_to_atom(match.group(1))
    right = _class_phrase_to_atom(match.group(2))
    if not left or not right:
        return None
    return Rule((left,), right, idx)


def _class_phrase_to_atom(phrase: str) -> Optional[Atom]:
    relational = re.match(r"^[A-Za-z]+\s+with\s+(.+)$", phrase.strip(), flags=re.I)
    if relational:
        phrase = f"has {relational.group(1)}"
    name, negated = _phrase_to_predicate(phrase)
    if not name:
        return None
    return Atom(name, CONTEXT_ENTITY, negated)


def _parse_who_rule(text: str, idx: int) -> List[Rule]:
    lowered = text.lower()
    if " who " not in lowered and " that " not in lowered:
        return []
    prefix = re.split(r"\s+(?:who|that)\s+", text, maxsplit=1, flags=re.I)[0]
    first_word = _tokens(prefix)[0] if _tokens(prefix) else ""
    if not re.match(r"^(every|all|any)\b", lowered) and not first_word.endswith("s"):
        return []

    split_match = re.match(r"^(.+?)\s+(?:who|that)\s+(.+)$", text, flags=re.I)
    if not split_match:
        return []
    domain = _strip_quantifier(_strip_articles(split_match.group(1))).strip()
    tail = split_match.group(2).strip()
    cond_text, conclusion_text = _split_rule_tail(tail)
    if not cond_text or not conclusion_text:
        return []

    conditions = [
        _clause_to_atom(clause, domain=domain)
        for clause in _split_conjunction(cond_text)
    ]
    conclusion = _clause_to_atom(conclusion_text, domain=domain)
    conditions = [atom for atom in conditions if atom]
    if not conditions or not conclusion:
        return []
    return [Rule(tuple(conditions), conclusion, idx)]


def _parse_participle_universal_rule(text: str, idx: int) -> Optional[Rule]:
    """Parse rules like "Every vendor receiving review appears in queue"."""

    if not re.match(r"^(?:every|all|any)\b", text, flags=re.I):
        return None

    match = re.match(
        r"^(?:every|all|any)\s+(.+?)\s+([A-Za-z]+ing\b.+)$",
        text,
        flags=re.I,
    )
    if not match:
        return None

    domain = _strip_quantifier(_strip_articles(match.group(1))).strip()
    tail = match.group(2).strip()
    cond_text, conclusion_text = _split_rule_tail(tail)
    if not domain or not cond_text or not conclusion_text:
        return None

    conditions = [
        _clause_to_atom(clause, domain=domain)
        for clause in _split_conjunction(cond_text)
    ]
    conclusion = _clause_to_atom(conclusion_text, domain=domain)
    conditions = [atom for atom in conditions if atom]
    if not conditions or not conclusion:
        return None
    return Rule(tuple(conditions), conclusion, idx)


def _ingest_fol_formula(fol: str, idx: int, state: ReasoningState) -> bool:
    normalized = _normalize_fol(fol)
    if not normalized:
        return False

    added = False
    for part in _split_top_level(normalized, "&"):
        part = _strip_outer(part.strip())
        if not part:
            continue

        rule = _parse_fol_rule(part, idx)
        if rule:
            state.rules.append(rule)
            added = True
            continue

        atom = _parse_fol_atom(part)
        if atom:
            _add_fact(state, atom, [idx])
            added = True
            continue

        numeric = _parse_fol_numeric(part, idx)
        if numeric:
            state.numeric_facts.append(numeric)
            _remember_entity(state, numeric.subject_key, numeric.subject_label)
            added = True

    return added


def _parse_fol_rule(fol: str, idx: int) -> Optional[Rule]:
    body = _unwrap_quantifier(fol)
    if "->" not in body:
        return None
    left, right = _split_once_top_level(body, "->")
    if not left or not right:
        return None
    antecedents = [
        _parse_fol_atom(piece.strip(), variable_entity=True)
        for piece in _split_top_level(_strip_outer(left), "&")
    ]
    consequent = _parse_fol_atom(_strip_outer(right).strip(), variable_entity=True)
    antecedents = [atom for atom in antecedents if atom]
    if not antecedents or not consequent:
        return None
    return Rule(tuple(antecedents), consequent, idx)


def _parse_fol_atom(text: str, variable_entity: bool = False) -> Optional[Atom]:
    text = _strip_outer(text.strip())
    if not text:
        return None
    negated = False
    if text.startswith("not "):
        negated = True
        text = text[4:].strip()
    if text.startswith("!"):
        negated = True
        text = text[1:].strip()
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\(([^(),]+)\)$", text)
    if not match:
        return None
    name = _normalize_predicate_name(match.group(1))
    arg = match.group(2).strip()
    entity = CONTEXT_ENTITY if variable_entity and _is_variable(arg) else _entity_key(arg)
    return Atom(name, entity, negated)


def _parse_fol_numeric(text: str, idx: int) -> Optional[NumericFact]:
    match = re.match(
        r"^([A-Za-z_][A-Za-z0-9_]*)\(([^(),]+)\)\s*(?:=|==)\s*([-+]?\d+(?:\.\d+)?)$",
        text.strip(),
    )
    if not match:
        return None
    subject = match.group(2).strip()
    return NumericFact(
        subject_key=_entity_key(subject),
        subject_label=subject,
        measure_key=_normalize_measure(match.group(1)),
        value=match.group(3),
        premise_index=idx,
    )


def _close_under_rules(state: ReasoningState) -> None:
    changed = True
    while changed:
        changed = False
        known_entities = set(state.entity_labels) or {CONTEXT_ENTITY}
        for atom in list(state.facts):
            known_entities.add(atom.entity)

        for rule in state.rules:
            candidate_entities = (
                known_entities
                if _rule_uses_context(rule)
                else {entity for entity in known_entities if entity != CONTEXT_ENTITY}
            )
            if not candidate_entities:
                candidate_entities = {CONTEXT_ENTITY}

            for entity in candidate_entities:
                antecedents = tuple(_bind_atom(atom, entity) for atom in rule.antecedents)
                consequent = _bind_atom(rule.consequent, entity)
                if all(atom in state.facts for atom in antecedents):
                    support = [rule.premise_index]
                    for atom in antecedents:
                        support.extend(state.facts.get(atom, ()))
                    if _add_fact(state, consequent, support):
                        changed = True

                # Safe classical contraposition for unary rules only:
                # A -> B and not B entail not A. For multi-antecedent rules,
                # not B does not identify which antecedent failed.
                if len(antecedents) == 1 and consequent.negate() in state.facts:
                    support = [rule.premise_index, *state.facts[consequent.negate()]]
                    if _add_fact(state, antecedents[0].negate(), support):
                        changed = True

        for clause in state.disjunctions:
            candidate_entities = {
                atom.entity
                for atom in state.facts
                if atom.entity != CONTEXT_ENTITY
            } or {CONTEXT_ENTITY}
            for entity in candidate_entities:
                alternatives = tuple(_bind_atom(atom, entity) for atom in clause.alternatives)
                for candidate in alternatives:
                    other_alternatives = [item for item in alternatives if item != candidate]
                    if other_alternatives and all(
                        item.negate() in state.facts for item in other_alternatives
                    ):
                        support = [clause.premise_index]
                        for item in other_alternatives:
                            support.extend(state.facts[item.negate()])
                        if _add_fact(state, candidate, support):
                            changed = True

        if not state.inconsistency_support:
            for requirement in state.existential_requirements:
                candidate_entities = {
                    atom.entity
                    for atom in state.facts
                    if atom.entity != CONTEXT_ENTITY
                }
                for entity in candidate_entities:
                    antecedent = _bind_atom(requirement.antecedent, entity)
                    antecedent_support = state.facts.get(antecedent)
                    if not antecedent_support:
                        continue
                    for absence in state.existential_absences:
                        if absence.entity != entity:
                            continue
                        if not _schema_names_match(
                            requirement.object_type,
                            absence.object_type,
                            state.domain_terms,
                        ):
                            continue
                        state.inconsistency_support = tuple(sorted({
                            requirement.premise_index,
                            absence.premise_index,
                            *antecedent_support,
                        }))
                        break
                    if state.inconsistency_support:
                        break
                if state.inconsistency_support:
                    break


def _answer_multiple_choice(
    question: str,
    options: Sequence[str],
    premises_nl: Sequence[str],
    state: ReasoningState,
    intent: str,
) -> Optional[PipelineResult]:
    parsed_options = _extract_options(question, options)
    if not parsed_options:
        return None

    if state.inconsistency_support:
        meta_options = [
            (letter, text)
            for letter, text in parsed_options
            if re.search(
                r"\b(?:logical\s+contradiction|inconsisten|unsat\s*core|"
                r"any\s+statement\s+provable)\b",
                text,
                flags=re.I,
            )
        ]
        if len(meta_options) == 1:
            letter, text = meta_options[0]
            used = state.inconsistency_support
            return PipelineResult(
                answer=letter,
                explanation=_explain_used(
                    f"Option {letter} is supported because the premise theory is inconsistent.",
                    premises_nl,
                    used,
                ),
                premises_used=list(used),
                confidence=0.99,
                query_type="type1",
                source="deterministic-symbolic-consistency",
                reasoning={
                    "type": "existential_consistency_check",
                    "status": "unsat",
                    "premises_used": list(used),
                },
                metadata={"logical_status": "Inconsistent"},
            )

    yes_no_labels = {"yes", "no", "uncertain"}
    if {text.lower() for _, text in parsed_options}.issubset(yes_no_labels):
        target = _question_to_atom(question, state.domain_terms)
        if target:
            verdict, used = _evaluate_atom_with_fol_abduction(target, state)
            if verdict is True:
                desired = "yes"
            elif verdict is False:
                desired = "no"
            else:
                desired = "uncertain"
            for letter, text in parsed_options:
                if text.lower() == desired:
                    explanation = _explain_used(
                        f"Option {letter} is supported: {text}.", premises_nl, used
                    )
                    return PipelineResult(
                        answer=letter,
                        explanation=explanation,
                        premises_used=list(used),
                        fol="\n".join(state.generated_fol) if state.generated_fol else None,
                        confidence=0.88,
                        query_type="type1",
                        source="deterministic-horn",
                    )

    candidates: List[Tuple[str, str, Tuple[int, ...]]] = []
    for letter, text in parsed_options:
        verdict, used = _evaluate_statement(text, state)
        if intent == "choose_false":
            if verdict is False:
                candidates.append((letter, text, used))
        elif verdict is True:
            candidates.append((letter, text, used))

    if not candidates:
        return None

    if intent == "choose_fewest_premises":
        candidates.sort(key=lambda item: (len(item[2]), item[0]))
    else:
        candidates.sort(key=lambda item: item[0])

    letter, text, used = candidates[0]
    explanation = _explain_used(
        f"Option {letter} is supported: {text}.", premises_nl, used
    )
    return PipelineResult(
        answer=letter,
        explanation=explanation,
        premises_used=list(used),
        fol="\n".join(state.generated_fol) if state.generated_fol else None,
        confidence=0.9,
        query_type="type1",
        source="deterministic-horn",
    )


def _answer_yes_no(
    question: str, premises_nl: Sequence[str], state: ReasoningState
) -> Optional[PipelineResult]:
    target = _question_to_atom(question, state.domain_terms)
    if not target:
        return None
    target = _resolve_atom(target, state)

    verdict, used = _evaluate_atom_with_fol_abduction(target, state)
    if verdict is True:
        answer = "Yes"
        explanation = _explain_used("The target is entailed by the premises.", premises_nl, used)
        confidence = 0.9
    elif verdict is False:
        answer = "No"
        explanation = _explain_used("The negation of the target is entailed by the premises.", premises_nl, used)
        confidence = 0.9
    else:
        note_idx = state.uncertainty_notes.get(target.positive())
        answer = "No" if _is_entailment_question(question) else "Uncertain"
        if target in state.facts and target.negate() in state.facts:
            answer = "Uncertain"
            used = tuple(sorted(set(state.facts[target]) | set(state.facts[target.negate()])))
            explanation = _explain_used(
                "The premises entail both the target and its negation, so no consistent verdict is available.",
                premises_nl,
                used,
            )
        else:
            used = (note_idx,) if note_idx is not None else used
            explanation = _explain_used(
                "The premises do not entail either the target or its negation.",
                premises_nl,
                used,
            )
        confidence = 0.88 if answer == "No" else 0.82

    return PipelineResult(
        answer=answer,
        explanation=explanation,
        premises_used=list(used),
        fol="\n".join(state.generated_fol) if state.generated_fol else None,
        confidence=confidence,
        query_type="type1",
        source="deterministic-horn",
    )


def _answer_open_ended(
    question: str, premises_nl: Sequence[str], state: ReasoningState
) -> Optional[PipelineResult]:
    numeric = _answer_numeric_question(question, premises_nl, state)
    if numeric is not None:
        return numeric

    predicate = _resolve_predicate_name(_question_to_open_predicate(question) or "", state)
    if not predicate:
        return None

    matches: List[Tuple[str, Tuple[int, ...]]] = []
    for atom, used in state.facts.items():
        if atom.negated or atom.entity == CONTEXT_ENTITY:
            continue
        if atom.name == predicate:
            label = state.entity_labels.get(atom.entity, atom.entity)
            matches.append((label, used))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0].lower())
    answer = ", ".join(label for label, _ in matches)
    used_indices = sorted({idx for _, used in matches for idx in used})
    explanation = _explain_used(
        f"The entity satisfying the requested property is {answer}.",
        premises_nl,
        tuple(used_indices),
    )
    return PipelineResult(
        answer=answer,
        explanation=explanation,
        premises_used=used_indices,
        fol="\n".join(state.generated_fol) if state.generated_fol else None,
        confidence=0.88,
        query_type="type1",
        source="deterministic-horn",
    )


def _answer_numeric_question(
    question: str, premises_nl: Sequence[str], state: ReasoningState
) -> Optional[PipelineResult]:
    derived = _answer_derived_value_question(question, premises_nl, state)
    if derived is not None:
        return derived

    grounded = _answer_grounded_value_question(question, premises_nl)
    if grounded is not None:
        return grounded

    match = re.match(
        r"^\s*how\s+many\s+(.+?)\s+(?:does|do|did)\s+(.+?)\s+(?:have|has|contain|include)\??\s*$",
        _clean_sentence(question),
        flags=re.I,
    )
    if not match:
        return None
    measure_key = _normalize_measure(match.group(1))
    subject_key = _entity_key(_strip_articles(match.group(2)))
    best = _find_numeric_fact(state.numeric_facts, subject_key, measure_key)
    if not best:
        return None
    explanation = _explain_used(
        f"The requested numeric value is directly stated as {best.value}.",
        premises_nl,
        (best.premise_index,),
    )
    return PipelineResult(
        answer=best.value,
        explanation=explanation,
        premises_used=[best.premise_index],
        confidence=0.94,
        query_type="type1",
        source="deterministic-horn",
    )


def _answer_derived_value_question(
    question: str, premises_nl: Sequence[str], state: ReasoningState
) -> Optional[PipelineResult]:
    """Project a scalar embedded in a proved attribute, preserving its proof."""

    cleaned = _clean_sentence(question)
    if not re.match(r"^(?:what|which|how many|in how many|at what time)\b", cleaned, flags=re.I):
        return None
    all_question_roots = {
        _normalize_symbol_token(token) for token in _tokens(cleaned)
    }
    question_roots = {
        _normalize_symbol_token(token)
        for token in _tokens(cleaned)
        if token not in {
            "what", "which", "how", "many", "is", "are", "does", "do",
            "did", "of", "the", "a", "an", "in", "at", "model",
        }
    }
    candidates: List[Tuple[int, int, Atom, Tuple[int, ...], str]] = []
    for atom, support in state.facts.items():
        if atom.negated or atom.entity == CONTEXT_ENTITY:
            continue
        label = state.entity_labels.get(atom.entity, atom.entity)
        label_roots = {_normalize_symbol_token(token) for token in _tokens(label)}
        if label_roots and not label_roots.issubset(all_question_roots):
            continue
        numbers = re.findall(r"(?:^|_)([-+]?\d+(?:\.\d+)?)(?:_|$)", atom.name)
        if not numbers:
            continue
        predicate_roots = {
            root for root in atom.name.split("_")
            if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", root)
            and root not in {"exactly", "initial", "current", "model"}
        }
        overlap = len(predicate_roots & question_roots)
        if overlap < 2:
            continue
        candidates.append((overlap, len(support), atom, support, numbers[-1]))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2].name))
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    if len({item[4] for item in best}) != 1:
        return None
    _, _, atom, support, value = best[0]
    unit = ""
    for candidate_unit in (
        "milliseconds", "seconds", "minutes", "hours", "percent",
        "megabytes", "gigabytes", "meters", "kilometers",
    ):
        if candidate_unit in cleaned.lower():
            unit = candidate_unit
            break
    return PipelineResult(
        answer=value,
        unit=unit,
        explanation=_explain_used(
            f"The requested value {value}{(' ' + unit) if unit else ''} is entailed by the rule chain.",
            premises_nl,
            support,
        ),
        premises_used=list(support),
        confidence=0.96,
        query_type="type1",
        source="deterministic-symbolic-projection",
        reasoning={
            "type": "derived_attribute_projection",
            "derived_predicate": atom.name,
            "premises_used": list(support),
            "value": value,
            "unit": unit,
        },
    )


def _answer_grounded_value_question(
    question: str, premises_nl: Sequence[str]
) -> Optional[PipelineResult]:
    """Project a typed scalar directly from the best-grounded premise.

    Values are selected by lexical grounding against all numeric/time-bearing
    premises. This avoids asking an SMT solver to manufacture a string or a
    number that is already explicitly present in the knowledge base.
    """

    cleaned = _clean_sentence(question)
    if not re.match(
        r"^(?:how many|in how many|what is|at what time)\b", cleaned, flags=re.I
    ):
        return None

    ignored = {
        "a", "an", "the", "of", "in", "at", "what", "which", "how", "many",
        "does", "do", "did", "is", "are", "was", "were", "this", "that",
    }
    query_tokens = {
        _normalize_symbol_token(token)
        for token in _tokens(cleaned)
        if token not in ignored and not token.isdigit()
    }
    value_pattern = re.compile(
        r"\b\d{1,2}:\d{2}\b|(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z])"
    )
    candidates: List[Tuple[int, int, str]] = []
    for idx, premise in enumerate(premises_nl):
        values = value_pattern.findall(str(premise))
        if not values:
            continue
        premise_tokens = {
            _normalize_symbol_token(token)
            for token in _tokens(premise)
            if token not in ignored and not token.isdigit()
        }
        overlap = len(query_tokens & premise_tokens)
        if overlap >= 2:
            candidates.append((overlap, idx, values[-1].replace(",", "")))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score = candidates[0][0]
    best = [candidate for candidate in candidates if candidate[0] == best_score]
    if len({value for _, _, value in best}) != 1:
        return None
    _, premise_index, value = best[0]
    return PipelineResult(
        answer=value,
        explanation=_explain_used(
            f"The requested value is explicitly grounded as {value}.",
            premises_nl,
            (premise_index,),
        ),
        premises_used=[premise_index],
        confidence=0.98,
        query_type="type1",
        source="deterministic-grounded-value",
        reasoning={
            "type": "attribute_projection",
            "premise_index": premise_index,
            "value": value,
        },
    )


def _evaluate_statement(text: str, state: ReasoningState) -> Tuple[Optional[bool], Tuple[int, ...]]:
    compound = _evaluate_compound_statement(text, state)
    if compound is not None:
        return compound
    numeric = _parse_numeric_fact(_clean_sentence(text), -1)
    if numeric:
        best = _find_numeric_fact(state.numeric_facts, numeric.subject_key, numeric.measure_key)
        if not best:
            return None, ()
        return best.value == numeric.value, (best.premise_index,)
    atom = _statement_to_atom(text, state.domain_terms)
    if not atom:
        return None, ()
    return _evaluate_atom(atom, state)


def _evaluate_compound_statement(
    text: str, state: ReasoningState
) -> Optional[Tuple[Optional[bool], Tuple[int, ...]]]:
    """Evaluate conjunctions and explicit proof-status clauses compositionally."""

    cleaned = _clean_sentence(text)
    pieces = [
        piece.strip(" ,")
        for piece in re.split(r"\s+(?:and|but)\s+", cleaned, flags=re.I)
        if piece.strip(" ,")
    ]
    if len(pieces) < 2:
        return None

    subject, _ = _extract_leading_entity(pieces[0], state.domain_terms)
    supports: set[int] = set()
    verdicts: List[Optional[bool]] = []
    for index, piece in enumerate(pieces):
        clause = piece
        if index and subject and not _extract_leading_entity(clause, state.domain_terms)[0]:
            clause = f"{subject} {clause}"

        not_entailed = re.match(
            r"^(.+?)\s+(?:is|are|was|were)\s+not\s+"
            r"(?:established|proved|proven|entailed|supported)"
            r"(?:\s+by\s+(?:the\s+)?premises?)?$",
            clause,
            flags=re.I,
        )
        if not_entailed:
            proposition = not_entailed.group(1).strip()
            if subject and not _extract_leading_entity(proposition, state.domain_terms)[0]:
                proposition = f"{subject} {proposition}"
            atom = _statement_to_atom(proposition, state.domain_terms)
            verdict, used = _evaluate_atom(atom, state) if atom else (None, ())
            verdicts.append(verdict is not True)
            supports.update(used)
            continue

        atom = _statement_to_atom(clause, state.domain_terms)
        verdict, used = _evaluate_atom(atom, state) if atom else (None, ())
        verdicts.append(verdict)
        supports.update(used)

    if all(verdict is True for verdict in verdicts):
        return True, tuple(sorted(supports))
    if any(verdict is False for verdict in verdicts):
        return False, tuple(sorted(supports))
    return None, tuple(sorted(supports))


def _evaluate_atom(atom: Atom, state: ReasoningState) -> Tuple[Optional[bool], Tuple[int, ...]]:
    atom = _resolve_atom(atom, state)
    opposite = atom.negate()
    if atom in state.facts and opposite in state.facts:
        support = tuple(sorted(set(state.facts[atom]) | set(state.facts[opposite])))
        return None, support
    if atom in state.facts:
        return True, state.facts[atom]
    if opposite in state.facts:
        return False, state.facts[opposite]
    return None, ()


def _evaluate_atom_with_fol_abduction(
    atom: Atom, state: ReasoningState
) -> Tuple[Optional[bool], Tuple[int, ...]]:
    atom = _resolve_atom(atom, state)
    verdict, used = _evaluate_atom(atom, state)
    if verdict is not None:
        return verdict, used

    # When only symbolic FOL is supplied, predicates can be opaque labels such
    # as C/M. If the question names the entity but not the opaque predicate, a
    # single newly derived positive/negative fact for that same entity is the
    # conservative target candidate. Ambiguous candidates are intentionally not
    # guessed.
    if atom.positive().name in _predicate_vocab(state):
        return None, ()

    consequent_names = {rule.consequent.name for rule in state.rules}
    positive_candidates = [
        (known, support)
        for known, support in state.facts.items()
        if (
            known.entity == atom.entity
            and not known.negated
            and known.name in consequent_names
            and len(support) > 1
        )
    ]
    negative_candidates = [
        (known, support)
        for known, support in state.facts.items()
        if (
            known.entity == atom.entity
            and known.negated
            and known.name in consequent_names
            and len(support) > 1
        )
    ]
    if len(positive_candidates) == 1 and not negative_candidates:
        return True, positive_candidates[0][1]
    if len(negative_candidates) == 1 and not positive_candidates:
        return False, negative_candidates[0][1]
    return None, ()


def _predicate_vocab(state: ReasoningState) -> set[str]:
    names = {atom.name for atom in state.facts}
    names.update(atom.name for atom in state.uncertainty_notes)
    for rule in state.rules:
        names.update(atom.name for atom in rule.antecedents)
        names.add(rule.consequent.name)
    return names


def _induce_predicate_schema(
    state: ReasoningState, learned_aliases: Optional[Mapping[str, str]] = None
) -> None:
    """Learn surface-form aliases from the predicates in the current problem.

    This is deliberately local and proof-independent: no benchmark vocabulary or
    answer labels are consulted. Two names are merged only when at least two
    normalized semantic roots overlap and one is largely an elaboration of the
    other (for example, ``created_account`` and
    ``completed_account_creation``).
    """

    names = sorted(_predicate_vocab(state))
    raw_learned = {
        _normalize_predicate_name(surface): _normalize_predicate_name(canonical)
        for surface, canonical in dict(learned_aliases or {}).items()
        if surface and canonical
    }
    # A global learned lexicon can contain thousands of short/generic aliases
    # (for example ``pass -> p``).  Injecting all of them into a local theory
    # makes query resolution ambiguous and can shorten proof support.  Only
    # ground local predicate surfaces through the learned schema; unseen query
    # wording is resolved against this scoped vocabulary below.
    learned = {
        surface: canonical
        for surface, canonical in raw_learned.items()
        if surface in names
    }
    grounded_names = {name: learned.get(name, name) for name in names}
    canonical_names = sorted(set(grounded_names.values()))
    parent = {name: name for name in canonical_names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for index, left in enumerate(canonical_names):
        for right in canonical_names[index + 1 :]:
            if _schema_names_match(left, right, state.domain_terms):
                union(left, right)

    groups: Dict[str, List[str]] = {}
    for name in canonical_names:
        groups.setdefault(find(name), []).append(name)

    canonical_aliases: Dict[str, str] = {}
    for members in groups.values():
        canonical = min(members, key=lambda value: (len(value.split("_")), len(value), value))
        canonical_aliases.update({member: canonical for member in members})

    aliases: Dict[str, str] = {
        surface: canonical_aliases.get(canonical, canonical)
        for surface, canonical in learned.items()
    }
    aliases.update(
        {
            surface: canonical_aliases.get(grounded, grounded)
            for surface, grounded in grounded_names.items()
        }
    )
    state.predicate_aliases = aliases

    def remap(atom: Atom) -> Atom:
        return Atom(aliases.get(atom.name, atom.name), atom.entity, atom.negated)

    remapped_facts: Dict[Atom, Tuple[int, ...]] = {}
    for atom, support in state.facts.items():
        mapped = remap(atom)
        previous = remapped_facts.get(mapped)
        if previous is None or (len(support), support) < (len(previous), previous):
            remapped_facts[mapped] = support
    state.facts = remapped_facts
    state.rules = [
        Rule(tuple(remap(atom) for atom in rule.antecedents), remap(rule.consequent), rule.premise_index)
        for rule in state.rules
    ]
    state.uncertainty_notes = {
        remap(atom): premise_index for atom, premise_index in state.uncertainty_notes.items()
    }


def _schema_names_match(
    left: str, right: str, domain_terms: Iterable[str] = ()
) -> bool:
    left_roots, right_roots = set(left.split("_")), set(right.split("_"))
    # Discourse connectives indicate a compound claim, not a paraphrase of one
    # of its clauses.  Without this guard, "safe because it contains data" can
    # collapse into the fact "contains data".
    discourse = {"and", "but", "becaus", "although", "unless", "while"}
    if bool(left_roots & discourse) != bool(right_roots & discourse):
        return False
    roman_numerals = {
        "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
        "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
    }
    number_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    }
    left_identifiers = {
        root for root in left_roots
        if root.isdigit() or root in roman_numerals or root in number_words
    }
    right_identifiers = {
        root for root in right_roots
        if root.isdigit() or root in roman_numerals or root in number_words
    }
    if left_identifiers != right_identifiers:
        return False
    shells = {
        "has", "have", "may", "must", "should", "will", "can", "becom",
        "issu", "receiv", "plac", "assign", "list", "appear", "that", "its",
    }
    local_domains = {_normalize_symbol_token(term) for term in domain_terms}
    semantic_left = left_roots - shells - local_domains
    semantic_right = right_roots - shells - local_domains
    if semantic_left and semantic_left == semantic_right:
        return True
    overlap = len(semantic_left & semantic_right)
    if (
        min(len(semantic_left), len(semantic_right)) == 1
        and overlap == 1
        and len(left_roots & right_roots) >= 3
    ):
        return True
    if overlap < 2:
        return False
    smaller = min(len(semantic_left), len(semantic_right))
    union_size = len(semantic_left | semantic_right)
    left_only = semantic_left - semantic_right
    right_only = semantic_right - semantic_left
    # Equal-shaped predicates with different content modifiers usually denote
    # different properties (budget approval vs supervisor approval, Physics I
    # vs Physics II).  Do not let a shared relational shell erase that detail.
    if len(left_only) == len(right_only) == 1:
        left_head = next(iter(left_only))
        right_head = next(iter(right_only))
        if SequenceMatcher(None, left_head, right_head).ratio() < 0.55:
            return False
    # Rule consequents and later antecedents often differ only by a domain role
    # or a light verb ("a stress alert is issued" / "has a stress alert").
    # Require two shared semantic roots, but tolerate those grammatical shells.
    if overlap / smaller >= 0.66 and overlap / union_size >= 0.4:
        return True
    if len(left_only) == len(right_only) == 1:
        left_head = next(iter(left_only))
        right_head = next(iter(right_only))
        return SequenceMatcher(None, left_head, right_head).ratio() >= 0.55
    return False


def _induce_directional_projections(state: ReasoningState) -> None:
    """Bridge a more specific consequent to a later, lexically weaker premise.

    Unlike aliasing this is directional: ``island_from_main_grid`` may satisfy
    ``island``, but observing the weaker predicate never fabricates the more
    specific one.
    """

    antecedents = {atom for rule in state.rules for atom in rule.antecedents}
    consequents = {rule.consequent for rule in state.rules}
    domains = {_normalize_symbol_token(term) for term in state.domain_terms}
    shells = {"has", "have", "may", "must", "should", "will", "can"}
    existing = {(rule.antecedents, rule.consequent) for rule in state.rules}
    additions: List[Rule] = []
    for specific in consequents:
        specific_roots = set(specific.name.split("_"))
        specific_semantic = specific_roots - domains - shells
        for general in antecedents:
            if specific.negated != general.negated:
                continue
            general_roots = set(general.name.split("_"))
            general_semantic = general_roots - domains - shells
            if not general_semantic or not general_semantic < specific_semantic:
                continue
            if len(specific_roots & general_roots) < 2:
                continue
            bridge = Rule((specific,), general, -1)
            if (bridge.antecedents, bridge.consequent) not in existing:
                additions.append(bridge)
                existing.add((bridge.antecedents, bridge.consequent))
    state.rules.extend(additions)


def _resolve_predicate_name(name: str, state: ReasoningState) -> str:
    if not name:
        return name
    direct = state.predicate_aliases.get(name)
    if direct:
        return direct
    matches = {
        canonical
        for candidate, canonical in state.predicate_aliases.items()
        if _schema_names_match(name, candidate, state.domain_terms)
    }
    return next(iter(matches)) if len(matches) == 1 else name


def _resolve_atom(atom: Atom, state: ReasoningState) -> Atom:
    return Atom(_resolve_predicate_name(atom.name, state), atom.entity, atom.negated)


def _question_to_atom(question: str, domain_terms: Iterable[str] = ()) -> Optional[Atom]:
    # Preserve newlines until the option block has been removed.  Cleaning
    # first used to collapse the block and turn the whole question into one
    # giant predicate.
    text = _clean_sentence(_remove_options(str(question or "")))
    text = re.sub(
        r",?\s*(?:according to|based on)\s+(?:the\s+)?premises?$",
        "",
        text,
        flags=re.I,
    ).strip(" ,")

    # Epistemic wrappers ask whether the embedded proposition is derivable;
    # they are not part of that proposition's predicate.
    wrapper = re.match(
        r"^(?:do|does|did)\s+(?:the\s+)?premises?\s+"
        r"(?:prove|establish|entail|show|guarantee|support)\s+that\s+(.+)$",
        text,
        flags=re.I,
    )
    if wrapper:
        text = wrapper.group(1).strip()

    text = re.sub(
        r"^(does|do|did)\s+(.+?)\s+(have|has)\s+",
        lambda m: f"{m.group(2)} has ",
        text,
        flags=re.I,
    )
    # Predicate-first passives such as "Is a zone recommended for Azure
    # Reef?" retain the trailing entity and merely drop the interrogative
    # auxiliary.  Entity-first questions are converted to declaratives.
    if re.match(r"^(?:is|are|was|were)\s+.+\s+for\s+", text, flags=re.I):
        text = re.sub(r"^(is|are|was|were)\s+", "", text, flags=re.I)
    else:
        interrogative = re.match(r"^(is|are|was|were)\s+(.+)$", text, flags=re.I)
        if interrogative:
            auxiliary, body = interrogative.groups()
            subject, remainder = _extract_leading_entity(body, domain_terms)
            if subject:
                text = f"{subject} {auxiliary.lower()} {remainder}"
            else:
                first, separator, rest = body.partition(" ")
                text = f"{first} {auxiliary.lower()} {rest}" if separator else body
    text = re.sub(
        r"^(can|may|must|should|will)\s+(.+?)\s+",
        lambda m: f"{m.group(2)} {m.group(1).lower()} ",
        text,
        flags=re.I,
    )
    text = re.sub(r"^(does|do|did)\s+", "", text, flags=re.I)
    return _statement_to_atom(text, domain_terms)


def _is_entailment_question(question: str) -> bool:
    """Whether the question asks about derivability rather than world truth."""

    text = _clean_sentence(_remove_options(str(question or ""))).lower()
    return bool(
        re.search(
            r"\b(?:premises?\s+(?:prove|establish|entail|show|support)|"
            r"guarantee|satisf(?:y|ies)\s+every\s+requirement)\b",
            text,
        )
    )


def _question_to_open_predicate(question: str) -> Optional[str]:
    text = _remove_options(_clean_sentence(question))
    match = re.match(r"^(?:which|what)\s+(.+?)\s+(.+)$", text, flags=re.I)
    if match:
        return _phrase_to_predicate(match.group(2))[0]
    match = re.match(r"^who\s+(.+)$", text, flags=re.I)
    if match:
        return _phrase_to_predicate(match.group(1))[0]
    return None


def _statement_to_atom(
    statement: str, domain_terms: Iterable[str] = ()
) -> Optional[Atom]:
    text = _clean_sentence(statement)
    if not text:
        return None

    truth_match = re.match(
        r"^([A-Za-z][A-Za-z0-9_]*)\s+(?:is|are)\s+(true|false)$",
        text,
        flags=re.I,
    )
    if truth_match:
        return Atom(
            _normalize_symbol_token(truth_match.group(1)),
            CONTEXT_ENTITY,
            truth_match.group(2).lower() == "false",
        )

    subject, remainder = _extract_leading_entity(text, domain_terms)
    if subject:
        name, negated = _phrase_to_predicate(remainder)
        if not name:
            return None
        entity_key = _entity_key(subject)
        return Atom(name, entity_key, negated)

    name, negated = _phrase_to_predicate(text)
    if not name:
        return None
    return Atom(name, CONTEXT_ENTITY, negated)


def _clause_to_atom(clause: str, domain: str = "") -> Optional[Atom]:
    text = _clean_sentence(clause)
    if not text:
        return None
    if domain:
        domain_words = r"\s+".join(re.escape(w) for w in _tokens(domain))
        text = re.sub(
            rf"^(?:that|the|a|an|any|every|all)?\s*{domain_words}\s+",
            "",
            text,
            flags=re.I,
        )
    name, negated = _phrase_to_predicate(text)
    if not name:
        return None
    return Atom(name, CONTEXT_ENTITY, negated)


def _phrase_to_predicate(phrase: str) -> Tuple[str, bool]:
    cleaned = _clean_sentence(phrase)
    truth_match = re.match(
        r"^([A-Za-z][A-Za-z0-9_]*)\s+(?:is|are)\s+(true|false)$",
        cleaned,
        flags=re.I,
    )
    if truth_match:
        return (
            _normalize_symbol_token(truth_match.group(1)),
            truth_match.group(2).lower() == "false",
        )
    if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", cleaned):
        return _normalize_symbol_token(cleaned), False

    text = cleaned.lower()
    # A hyphenated lexical compound ("no-take", "no-fly") is not logical
    # negation.  Protect it before processing standalone negative markers.
    text = re.sub(r"\bno-([a-z])", r"no\1", text)
    negated = False

    negative_patterns = [
        (r"\bcannot\b", "can"),
        (r"\bcan\s+not\b", "can"),
        (r"\bdoes\s+not\s+have\b", "has"),
        (r"\bdo\s+not\s+have\b", "has"),
        (r"\bdid\s+not\s+have\b", "has"),
        (r"\bhas\s+not\b", "has"),
        (r"\bhave\s+not\b", "has"),
        (r"\bis\s+not\b", "is"),
        (r"\bare\s+not\b", "are"),
        (r"\blacks?\b", "has"),
        (r"\bfail(?:s|ed)?\b", "passes"),
        (r"\b(?:does|do|did)\s+not\b", ""),
        (r"\bnever\b", ""),
        (r"\bnot\b", ""),
        (r"\bno\b", ""),
    ]
    for pattern, replacement in negative_patterns:
        if re.search(pattern, text):
            negated = True
            text = re.sub(pattern, replacement, text)

    text = re.sub(r"\b(it|its|this|that)\s+", "", text)
    text = re.sub(r"\btoday\b|\bnow\b|\bcurrently\b", "", text)
    text = re.sub(r"\bis\s+raining\b", "rains", text)
    text = re.sub(r"\brain(?:ing)?\b", "rains", text)
    text = re.sub(r"\b(gets|get|got|becomes|become|is|are)\s+wet\b", "wet", text)
    text = re.sub(r"\b(is|are)\s+dry\b", "dry", text)
    text = re.sub(r"\b(is|are|was|were|be|being|been)\b", " ", text)
    text = re.sub(r"\b(has|have)\s+([a-z]+ed)\b", r"\2", text)
    text = re.sub(r"\b(has|have)\s+(passed|received|awarded|completed)\b", r"\2", text)
    text = text.replace("'s", "")
    text = text.replace("-", " ")
    text = _strip_articles(text)

    tokens = _tokens(text)
    stop = {
        "a",
        "an",
        "the",
        "that",
        "this",
        "same",
        "as",
        "to",
        "for",
        "of",
        "in",
        "on",
        "with",
        "who",
        "which",
        "what",
    }
    normalized: List[str] = []
    for token in tokens:
        if token in stop:
            continue
        if token == "have":
            token = "has"
        elif token == "can":
            token = "may"
        normalized.append(_normalize_symbol_token(token))
    return "_".join(normalized), negated


def _extract_leading_entity(
    text: str, domain_terms: Iterable[str] = ()
) -> Tuple[str, str]:
    text = text.strip()

    # Relational possessives often name the concrete entity after ``of``:
    # "The heatwave sensor of Greenhouse Basil is active".  Preserve the
    # property head while grounding the fact on the named entity.
    of_entity = re.match(
        r"^(?:the\s+)?(.+?)\s+of\s+"
        r"([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*)\s+"
        r"(is|are|was|were|has|have|can|may|must|should|will)\s+(.+)$",
        text,
    )
    if of_entity:
        property_head, entity, auxiliary, remainder = of_entity.groups()
        return entity.strip(), f"{property_head} {auxiliary} {remainder}".strip()

    # Passive and relational facts often place the concrete entity at the end:
    # "An emergency waiver is approved for MedKit-7".  Ground that entity and
    # keep the entire preceding property phrase.
    trailing = re.match(
        r"^(.+?)\s+for\s+((?:[A-Z][A-Za-z0-9_-]*)(?:\s+[A-Z][A-Za-z0-9_-]*)*)"
        r"(?:\s+(is|are|was|were)\s+(.+))?$",
        text,
    )
    if trailing:
        property_text = re.sub(r"^(?:the|a|an)\s+", "", trailing.group(1), flags=re.I)
        if trailing.group(4):
            property_text = f"{property_text} {trailing.group(3)} {trailing.group(4)}"
        return trailing.group(2).strip(), property_text.strip()

    # Strip a determiner only when it precedes a proper-name sequence.  The old
    # implementation discarded every "The ..." subject and leaked entity names
    # into predicate strings.
    candidate = re.sub(r"^the\s+", "", text, flags=re.I)

    # "All affected Atlas passwords ..." is a grounded statement about Atlas,
    # not a universal class rule.  Preserve the surrounding role words as part
    # of the predicate while extracting the proper name.
    quantified_named = re.match(
        r"^(?i:all|every|any)\s+(.+?)\s+([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*)\s+(.+)$",
        text,
    )
    if quantified_named:
        prefix, entity, suffix = quantified_named.groups()
        if re.search(
            r"\b(?:has|have|is|are|was|were|can|may|must|should|will|"
            r"[A-Za-z]+(?:ed|ing|es|s))\b",
            suffix,
            flags=re.I,
        ):
            return entity.strip(), f"{prefix} {suffix}".strip()

    match = re.match(
        r"^([A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*)\s+(.+)$",
        candidate,
    )
    if not match:
        return "", text
    subject = match.group(1).strip()
    remainder = match.group(2).strip()
    # Acronyms following a proper name are normally an attribute head ("River
    # Codex OCR ..."), not part of the entity.  Move them back to the property.
    subject_parts = subject.split()
    while len(subject_parts) > 1 and re.fullmatch(r"[A-Z]{2,}", subject_parts[-1]):
        remainder = f"{subject_parts.pop()} {remainder}"
    subject = " ".join(subject_parts)
    # Avoid treating sentence-initial common words as entities.
    if subject.lower() in {
        "if", "every", "all", "no", "which", "what", "who", "based",
        "an", "a", "the", "it", "this", "that",
    }:
        return "", text
    return subject, remainder


def _infer_domain(conditions: str, conclusion: str) -> str:
    """Infer the rule variable phrase from repetition, not a domain vocabulary."""

    raw_left = conditions.strip()
    left_for_common = re.sub(
        r"^(?:a|an|the|any|every|all)\s+", "", raw_left, flags=re.I
    )
    right = re.sub(
        r"^(?:then\s+)?(?:it|that|the|same)\s+", "", conclusion, flags=re.I
    )
    common: List[str] = []
    for left_token, right_token in zip(_tokens(left_for_common), _tokens(right)):
        if _normalize_symbol_token(left_token) != _normalize_symbol_token(right_token):
            break
        if common and left_token.lower() in {
            "is", "are", "was", "were", "can", "may", "must", "should", "will"
        }:
            break
        common.append(left_token)
    if common:
        return " ".join(common)

    quantified = re.match(r"^(?:a|an|the|any|every|all)\s+(.+)$", raw_left, flags=re.I)
    if quantified:
        first_clause = _split_conjunction(quantified.group(1))[0]
        words = first_clause.split()
        auxiliaries = {
            "is", "are", "was", "were", "has", "have", "can", "may",
            "must", "should", "will", "does", "do", "did",
        }
        for index, word in enumerate(words[1:], start=1):
            token = re.sub(r"[^A-Za-z]", "", word).lower()
            if token in auxiliaries or _looks_like_predicate_head(token):
                return " ".join(words[:index])

    return ""


def _infer_rule_domain(text: str) -> set[str]:
    match = re.match(r"^if\s+(.+?)(?:,\s*)?then\s+(.+)$", text, flags=re.I)
    if not match:
        return set()
    return set(_tokens(_infer_domain(match.group(1), match.group(2))))


def _expand_elliptical_conjuncts(text: str, domain: str) -> List[str]:
    """Restore a shared predicate head in phrases such as ``passed X and Y``."""

    clauses = _split_conjunction(text)
    if len(clauses) < 2:
        return clauses
    first = clauses[0]
    if domain:
        first = re.sub(
            rf"^(?:a|an|the|any|every|all)?\s*{re.escape(domain)}\s+",
            "",
            first,
            flags=re.I,
        )
    first_words = first.split()
    if not first_words:
        return clauses
    shared_head = first_words[0]
    expanded = [clauses[0]]
    for clause in clauses[1:]:
        first_token = clause.split()[0] if clause.split() else ""
        if first_token[:1].isupper():
            clause = f"{shared_head} {clause}"
        expanded.append(clause)
    return expanded


def _ingest_deferred_universals(state: ReasoningState) -> None:
    pending = list(state.pending_universals)
    state.pending_universals.clear()
    while pending:
        vocab = _predicate_vocab(state)
        progress = False
        remaining: List[Tuple[str, int]] = []
        for text, idx in pending:
            body = re.sub(r"^(?:all|every|any)\s+", "", text, flags=re.I)
            words = body.split()
            selected: Optional[Rule] = None
            for split in range(1, len(words)):
                left = _class_phrase_to_atom(" ".join(words[:split]))
                right = _class_phrase_to_atom(" ".join(words[split:]))
                if not left or not right:
                    continue
                if left.name in vocab:
                    selected = Rule((left,), right, idx)
                    break
                object_matches = _object_schema_matches(left.name, vocab)
                if len(object_matches) == 1:
                    selected = Rule(
                        (Atom(object_matches[0], left.entity, left.negated),), right, idx
                    )
                    break
                if selected is None and any(
                    _schema_names_match(left.name, known) for known in vocab
                ):
                    selected = Rule((left,), right, idx)
            if selected:
                state.rules.append(selected)
                progress = True
            else:
                remaining.append((text, idx))
        if not progress:
            state.pending_universals.extend(remaining)
            break
        pending = remaining


def _object_schema_matches(name: str, vocabulary: Iterable[str]) -> List[str]:
    """Match a relational class by its unique object signature.

    For example, ``people with lounge access`` is represented as
    ``has_lounge_access`` and can bind to the already observed
    ``receive_lounge_access`` predicate without a verb synonym table.
    """

    parts = name.split("_")
    if len(parts) < 3:
        return []
    object_roots = set(parts[1:])
    if len(object_roots) < 2:
        return []
    return sorted(
        candidate
        for candidate in vocabulary
        if object_roots.issubset(set(candidate.split("_")[1:]))
    )


def _split_rule_tail(tail: str) -> Tuple[str, str]:
    words = list(re.finditer(r"[A-Za-z]+", tail))
    grammatical = [
        match
        for match in words[1:]
        if match.group(0).lower()
        in {"is", "are", "was", "were", "can", "may", "must", "should", "will"}
    ]
    candidates = [
        match for match in words[1:] if _looks_like_predicate_head(match.group(0))
    ]
    if not grammatical and not candidates:
        return "", ""
    boundary = grammatical[-1] if grammatical else candidates[-1]
    return tail[: boundary.start()].strip(), tail[boundary.start() :].strip()


def _looks_like_predicate_head(token: str) -> bool:
    lowered = token.lower()
    if lowered in {"is", "are", "was", "were", "can", "may", "must", "should", "will"}:
        return True
    return len(lowered) > 3 and lowered.endswith(("ing", "ed", "es", "s"))


def _strip_quantifier(text: str) -> str:
    return re.sub(r"^(?:every|all|any|each)\s+", "", str(text or "").strip(), flags=re.I)


def _split_conjunction(text: str) -> List[str]:
    # Do not split numeric ranges such as "between 2 and 8 degrees".
    protected = re.sub(
        r"\bbetween\s+([^,]+?)\s+and\s+([^,]+?)(?=\s+(?:and|,|then)\b|$)",
        lambda m: f"between {m.group(1)} __RANGE_AND__ {m.group(2)}",
        text,
        flags=re.I,
    )
    return [
        part.replace("__RANGE_AND__", "and").strip(" ,")
        for part in re.split(r"\s+(?:and|&)\s+", protected, flags=re.I)
        if part.strip(" ,")
    ]


def _extract_options(question: str, options: Sequence[str]) -> List[Tuple[str, str]]:
    lettered = re.findall(
        r"(?:^|\n)\s*([A-Z])[\.)]\s*(.+?)(?=\n\s*[A-Z][\.)]\s*|\Z)",
        question,
        flags=re.DOTALL,
    )
    if lettered:
        return [(letter, _clean_sentence(text)) for letter, text in lettered]

    parsed: List[Tuple[str, str]] = []
    for idx, option in enumerate(options or []):
        text = str(option).strip()
        if not text or re.fullmatch(r"[A-Z]", text):
            continue
        match = re.match(r"^([A-Z])[\.)]\s*(.+)$", text)
        if match:
            parsed.append((match.group(1), _clean_sentence(match.group(2))))
        else:
            parsed.append((chr(65 + idx), _clean_sentence(text)))
    return parsed


def _remove_options(question: str) -> str:
    return re.sub(
        r"(?:^|\n)\s*[A-Z][\.)]\s*.+?(?=\n\s*[A-Z][\.)]\s*|\Z)",
        "",
        question,
        flags=re.DOTALL,
    ).strip()


def _find_numeric_fact(
    facts: Sequence[NumericFact], subject_key: str, measure_key: str
) -> Optional[NumericFact]:
    measure_tokens = set(measure_key.split("_"))
    best: Optional[Tuple[int, NumericFact]] = None
    for fact in facts:
        if fact.subject_key != subject_key:
            continue
        fact_tokens = set(fact.measure_key.split("_"))
        overlap = len(measure_tokens & fact_tokens)
        if overlap == 0:
            continue
        score = overlap * 2 - abs(len(fact_tokens) - len(measure_tokens))
        if best is None or score > best[0]:
            best = (score, fact)
    return best[1] if best else None


def _add_fact(state: ReasoningState, atom: Atom, support: Iterable[int]) -> bool:
    clean_support = tuple(sorted({idx for idx in support if idx >= 0}))
    existing = state.facts.get(atom)
    if existing is not None:
        if not clean_support:
            return False
        if existing and (len(existing), existing) <= (len(clean_support), clean_support):
            return False
        state.facts[atom] = clean_support
        return True
    state.facts[atom] = clean_support
    _remember_entity(state, atom.entity, atom.entity)
    return True


def _remember_entity(state: ReasoningState, entity_key: str, label: str) -> None:
    if not entity_key or entity_key == CONTEXT_ENTITY:
        return
    if entity_key not in state.entity_labels:
        state.entity_labels[entity_key] = _display_entity(label)


def _bind_atom(atom: Atom, entity: str) -> Atom:
    if atom.entity == CONTEXT_ENTITY:
        return Atom(atom.name, entity, atom.negated)
    return atom


def _rule_uses_context(rule: Rule) -> bool:
    return all(atom.entity == CONTEXT_ENTITY for atom in (*rule.antecedents, rule.consequent))


def _normalize_fol(fol: str) -> str:
    text = fol.strip()
    text = text.replace("\u00ac", "not ")
    text = text.replace("~", "not ")
    text = text.replace("\u2227", "&")
    text = text.replace("\u2228", "|")
    text = text.replace("\u2192", "->")
    text = text.replace("\u2265", ">=")
    text = text.replace("\u2264", "<=")
    text = re.sub(r"\bForAll\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(.+)\)\s*$", r"forall \1 (\2)", text)
    text = re.sub(r"^\u2200\s*([A-Za-z_][A-Za-z0-9_]*)\s*(.+)$", r"forall \1 \2", text)
    return text


def _unwrap_quantifier(text: str) -> str:
    text = text.strip()
    while True:
        match = re.match(r"^forall\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.+)$", text, flags=re.I)
        if not match:
            return _strip_outer(text)
        text = _strip_outer(match.group(2).strip())


def _split_once_top_level(text: str, operator: str) -> Tuple[str, str]:
    depth = 0
    for idx in range(len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and text.startswith(operator, idx):
            return text[:idx].strip(), text[idx + len(operator) :].strip()
    return "", ""


def _split_top_level(text: str, operator: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and text.startswith(operator, idx):
            parts.append(text[start:idx].strip())
            start = idx + len(operator)
            idx = start
            continue
        idx += 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _strip_outer(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")") and _balanced(text[1:-1]):
        text = text[1:-1].strip()
    return text


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _normalize_predicate_name(name: str) -> str:
    return "_".join(_normalize_symbol_token(token) for token in _tokens(name))


def _normalize_symbol_token(token: str) -> str:
    token = str(token or "").strip().lower()
    if not token:
        return token
    if token in {"is", "are", "was", "were", "has", "does", "this"}:
        return token
    stripped_inflection = False
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ation"):
        token = token[:-3]
    elif len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
        stripped_inflection = True
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
        stripped_inflection = True
        if token.endswith("i"):
            token = token[:-1] + "y"
    elif len(token) > 4 and token.endswith("sses"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    if stripped_inflection and len(token) > 3 and token[-1] == token[-2] and not token.endswith("ss"):
        token = token[:-1]
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def _normalize_measure(text: str) -> str:
    text = _strip_articles(_clean_sentence(text).lower())
    text = re.sub(r"\b(total|number|amount|count|of|the|a|an)\b", " ", text)
    return "_".join(_tokens(text))


def _entity_key(text: str) -> str:
    return "_".join(_tokens(str(text).lower())) or CONTEXT_ENTITY


def _display_entity(text: str) -> str:
    raw = str(text).strip()
    if not raw:
        return raw
    if raw == raw.lower() and "_" in raw:
        return " ".join(part.capitalize() for part in raw.split("_"))
    return raw.strip("_")


def _strip_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.I).strip()


def _clean_sentence(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^[A-Z][\.)]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .?;:\n\t")


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def _is_variable(text: str) -> bool:
    return bool(re.fullmatch(r"[a-z]", text.strip()))


def _explain_used(prefix: str, premises_nl: Sequence[str], used: Tuple[int, ...]) -> str:
    used = tuple(sorted({idx for idx in used if 0 <= idx < len(premises_nl)}))
    if not used:
        return prefix
    cited = "; ".join(f"P{idx + 1}: {premises_nl[idx]}" for idx in used)
    return f"{prefix} Used premises: {cited}"
