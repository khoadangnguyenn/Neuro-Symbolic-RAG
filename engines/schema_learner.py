"""Learn a predicate lexicon from aligned natural-language/FOL training data.

The learner deliberately ignores questions, answers, and explanations.  Its
only supervision is the position and role of predicates in aligned NL/FOL
premises.  The resulting aliases ground surface language into a stable
symbolic vocabulary; the Horn/Z3 layers still have to prove every answer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from exact_pipeline.engines.horn_reasoner import ReasoningState, build_reasoning_state


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PredicateEvidence:
    canonical: str
    support: int
    total: int

    @property
    def confidence(self) -> float:
        return self.support / self.total if self.total else 0.0


class PredicateSchemaLearner:
    """Induce and optionally persist a confidence-filtered predicate schema."""

    def __init__(self, aliases: Mapping[str, str], evidence: Mapping[str, PredicateEvidence]) -> None:
        self.aliases = dict(aliases)
        self.evidence = dict(evidence)

    @classmethod
    def from_examples(
        cls,
        examples: Sequence[object],
        *,
        cache_path: Optional[Path] = None,
        min_confidence: float = 0.8,
    ) -> "PredicateSchemaLearner":
        pairs = tuple(_unique_aligned_pairs(examples))
        fingerprint = _fingerprint(pairs, min_confidence=min_confidence)
        if cache_path:
            cached = _load_cache(cache_path, fingerprint)
            if cached is not None:
                return cached

        counts: Dict[str, Counter[str]] = defaultdict(Counter)
        for nl, fol in pairs:
            for surface, canonical in _aligned_predicates(nl, fol):
                if surface and canonical:
                    counts[surface][canonical] += 1

        aliases: Dict[str, str] = {}
        evidence: Dict[str, PredicateEvidence] = {}
        for surface, candidates in counts.items():
            ranked = candidates.most_common()
            if not ranked:
                continue
            canonical, support = ranked[0]
            total = sum(candidates.values())
            confidence = support / total
            # A tied mapping is semantically ambiguous even when its numeric
            # confidence happens to pass a permissive threshold.
            tied = len(ranked) > 1 and ranked[1][1] == support
            if not tied and confidence >= min_confidence:
                aliases[surface] = canonical
                aliases.setdefault(canonical, canonical)
                evidence[surface] = PredicateEvidence(canonical, support, total)

        learner = cls(aliases, evidence)
        if cache_path:
            _save_cache(cache_path, fingerprint, learner)
        return learner

    def metadata(self) -> dict:
        return {
            "version": SCHEMA_VERSION,
            "aliases": len(self.aliases),
            "learned_surfaces": len(self.evidence),
        }


def _unique_aligned_pairs(examples: Sequence[object]) -> Iterable[Tuple[str, str]]:
    seen = set()
    for example in examples:
        premises_nl = tuple(getattr(example, "premises_nl", ()) or ())
        premises_fol = tuple(getattr(example, "premises_fol", ()) or ())
        if len(premises_nl) != len(premises_fol):
            continue
        for nl, fol in zip(premises_nl, premises_fol):
            pair = (str(nl).strip(), str(fol).strip())
            if not all(pair) or pair in seen:
                continue
            seen.add(pair)
            yield pair


def _aligned_predicates(nl: str, fol: str) -> Iterable[Tuple[str, str]]:
    nl_state = build_reasoning_state([nl], [])
    fol_state = build_reasoning_state([], [fol])

    if len(nl_state.rules) == len(fol_state.rules) == 1:
        nl_rule, fol_rule = nl_state.rules[0], fol_state.rules[0]
        if len(nl_rule.antecedents) == len(fol_rule.antecedents):
            for nl_atom, fol_atom in zip(nl_rule.antecedents, fol_rule.antecedents):
                if nl_atom.negated == fol_atom.negated:
                    yield nl_atom.name, fol_atom.name
            if nl_rule.consequent.negated == fol_rule.consequent.negated:
                yield nl_rule.consequent.name, fol_rule.consequent.name
        return

    nl_facts = _ordered_facts(nl_state)
    fol_facts = _ordered_facts(fol_state)
    if len(nl_facts) == len(fol_facts) == 1 and nl_facts[0][1] == fol_facts[0][1]:
        yield nl_facts[0][0], fol_facts[0][0]


def _ordered_facts(state: ReasoningState) -> Tuple[Tuple[str, bool], ...]:
    return tuple((atom.name, atom.negated) for atom in state.facts)


def _fingerprint(
    pairs: Sequence[Tuple[str, str]], *, min_confidence: float
) -> str:
    digest = sha256()
    digest.update(
        f"schema-v{SCHEMA_VERSION};min-confidence={min_confidence:.6f}\n".encode()
    )
    for nl, fol in pairs:
        digest.update(nl.encode("utf-8"))
        digest.update(b"\0")
        digest.update(fol.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_cache(path: Path, fingerprint: str) -> Optional[PredicateSchemaLearner]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("version") != SCHEMA_VERSION or raw.get("fingerprint") != fingerprint:
            return None
        aliases = raw.get("aliases")
        evidence_raw = raw.get("evidence")
        if not isinstance(aliases, dict) or not isinstance(evidence_raw, dict):
            return None
        evidence = {
            surface: PredicateEvidence(
                canonical=str(item["canonical"]),
                support=int(item["support"]),
                total=int(item["total"]),
            )
            for surface, item in evidence_raw.items()
            if isinstance(item, dict)
        }
        return PredicateSchemaLearner(aliases, evidence)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _save_cache(path: Path, fingerprint: str, learner: PredicateSchemaLearner) -> None:
    payload = {
        "version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "aliases": learner.aliases,
        "evidence": {
            surface: {
                "canonical": item.canonical,
                "support": item.support,
                "total": item.total,
                "confidence": item.confidence,
            }
            for surface, item in learner.evidence.items()
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        # A read-only deployment should still be able to learn in memory.
        return
