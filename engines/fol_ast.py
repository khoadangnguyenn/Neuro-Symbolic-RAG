"""Validated JSON AST for solver-bound first-order logic.

The language model is allowed to choose the semantics, but never the concrete
solver syntax.  This module validates scope/arity/polarity and deterministically
compiles a small, compositional FOL AST into the grammar consumed by Z3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import re
import copy


class AstValidationError(ValueError):
    pass


# Resource/syntax guards. These bound representation size, not problem
# semantics: larger formulas can always be expressed with nested quantifiers
# and connectives.
MAX_IDENTIFIER_LENGTH = 128
MAX_TERMS_PER_ATOM = 16
MAX_VARS_PER_QUANTIFIER = 16
MAX_CONNECTIVE_CHILDREN = 32
MAX_AST_NODES = 512
MAX_AST_DEPTH = 64
MAX_PREFIX_TOKENS = 512


@dataclass
class AstCompileContext:
    predicate_arities: Dict[str, int] = field(default_factory=dict)
    constants: Set[str] = field(default_factory=set)
    nodes_seen: int = 0


@dataclass(frozen=True)
class CompiledAstTranslation:
    premises_fol: Tuple[str, ...]
    target_fol: str
    options_fol: Tuple[str, ...]
    predicates: Tuple[str, ...]
    constants: Tuple[str, ...]
    entity_mentions: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    predicate_mentions: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()


_OPS = {
    "atom", "not", "and", "or", "implies", "iff", "forall", "exists",
    "true", "false", "none", "theory_inconsistent",
}


def coerce_translation_envelope(raw: object) -> Mapping[str, object]:
    """Recover a uniquely identifiable translation object from shallow wrappers."""
    if isinstance(raw, Mapping) and isinstance(raw.get("translation"), Mapping):
        return raw

    candidates: List[Mapping[str, object]] = []

    def visit(value: object, depth: int) -> None:
        if depth > 3:
            return
        if isinstance(value, Mapping):
            keys = set(value)
            compact = {"premises", "target_fol", "options_fol"}.issubset(keys)
            ast = {"premises", "target", "options"}.issubset(keys)
            prefix = {"premises", "target_prefix", "options_prefix"}.issubset(keys)
            if compact or ast or prefix:
                candidates.append(value)
                return
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth + 1)

    visit(raw, 0)
    unique = []
    seen = set()
    for candidate in candidates:
        marker = id(candidate)
        if marker not in seen:
            seen.add(marker)
            unique.append(candidate)
    if len(unique) != 1:
        raise AstValidationError(
            f"expected one translation payload, found {len(unique)}"
        )
    return {"translation": unique[0]}


def logic_ast_json_schema() -> dict:
    """Structured-output schema; semantic constraints are checked in Python."""

    expression = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": sorted(_OPS)},
            "predicate": {"type": "string", "minLength": 1, "maxLength": MAX_IDENTIFIER_LENGTH},
            "terms": {
                "type": "array", "minItems": 1, "maxItems": MAX_TERMS_PER_ATOM,
                "items": {"type": "string", "minLength": 1, "maxLength": MAX_IDENTIFIER_LENGTH},
            },
            "args": {
                "type": "array", "minItems": 1, "maxItems": MAX_CONNECTIVE_CHILDREN,
                "items": {"$ref": "#/$defs/expression"},
            },
            "left": {"$ref": "#/$defs/expression"},
            "right": {"$ref": "#/$defs/expression"},
            "body": {"$ref": "#/$defs/expression"},
            "vars": {
                "type": "array", "minItems": 1, "maxItems": MAX_VARS_PER_QUANTIFIER,
                "items": {"type": "string", "minLength": 1, "maxLength": MAX_IDENTIFIER_LENGTH},
            },
        },
        "required": ["op"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "$defs": {"expression": expression},
        "properties": {
            "translation": {
                "type": "object",
                "properties": {
                    "entities": {"type": "array"},
                    "predicates": {"type": "array"},
                    "premises": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_index": {"type": "integer", "minimum": 0},
                                "ast": {"$ref": "#/$defs/expression"},
                            },
                            "required": ["source_index", "ast"],
                            "additionalProperties": False,
                        },
                    },
                    "target": {"$ref": "#/$defs/expression"},
                    "options": {"type": "array", "items": {"$ref": "#/$defs/expression"}},
                },
                "required": ["premises", "target", "options"],
                "additionalProperties": False,
            }
        },
        "required": ["translation"],
        "additionalProperties": False,
    }


def logic_flat_ast_json_schema() -> dict:
    """Non-recursive wire schema for constrained decoders such as Qwen.

    Nodes refer to child node IDs. Python reconstructs the recursive typed AST
    after validating IDs, reachability, cycles, and operator arity.
    """
    node = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "minimum": 0, "maximum": MAX_AST_NODES - 1},
            "op": {"type": "string", "enum": sorted(_OPS)},
            "predicate": {
                "type": "string",
                "pattern": "^p[0-9]+$",
                "maxLength": MAX_IDENTIFIER_LENGTH,
            },
            "terms": {
                "type": "array", "maxItems": MAX_TERMS_PER_ATOM,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_IDENTIFIER_LENGTH,
                },
            },
            "children": {
                "type": "array", "maxItems": MAX_CONNECTIVE_CHILDREN,
                "items": {"type": "integer", "minimum": 0, "maximum": MAX_AST_NODES - 1},
            },
            "vars": {
                "type": "array", "maxItems": MAX_VARS_PER_QUANTIFIER,
                "items": {"type": "string", "maxLength": MAX_IDENTIFIER_LENGTH},
            },
        },
        "required": ["id", "op"],
        "additionalProperties": False,
    }
    formula = {
        "type": "object",
        "properties": {
            "root": {"type": "integer", "minimum": 0, "maximum": MAX_AST_NODES - 1},
            "nodes": {"type": "array", "minItems": 1, "maxItems": MAX_AST_NODES, "items": node},
        },
        "required": ["root", "nodes"],
        "additionalProperties": False,
    }
    declaration = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^e[0-9]+$"},
            "mentions": {
                "type": "array", "minItems": 1, "maxItems": 16,
                "items": {"type": "string", "minLength": 1, "maxLength": 256},
            },
        },
        "required": ["id", "mentions"],
        "additionalProperties": False,
    }
    predicate_declaration = {
        "type": "object",
        "properties": {
            **declaration["properties"],
            "id": {"type": "string", "pattern": "^p[0-9]+$"},
            "arity": {"type": "integer", "minimum": 1, "maximum": MAX_TERMS_PER_ATOM},
        },
        "required": ["id", "mentions", "arity"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "translation": {
                "type": "object",
                "properties": {
                    "entities": {"type": "array", "maxItems": 128, "items": declaration},
                    "predicates": {"type": "array", "maxItems": 128, "items": predicate_declaration},
                    "premises": {
                        "type": "array", "maxItems": 256,
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_index": {"type": "integer", "minimum": 0},
                                "formula": formula,
                            },
                            "required": ["source_index", "formula"],
                            "additionalProperties": False,
                        },
                    },
                    "target": formula,
                    "options": {"type": "array", "maxItems": 64, "items": formula},
                },
                "required": ["entities", "predicates", "premises", "target", "options"],
                "additionalProperties": False,
            }
        },
        "required": ["translation"],
        "additionalProperties": False,
    }


def logic_prefix_ir_json_schema() -> dict:
    """Compact non-recursive schema for grammar-constrained semantic parsing.

    Formula trees are prefix token streams. Operator arity and predicate arity
    make the stream unambiguous, while keeping generation much shorter than a
    node graph. Python reconstructs and validates the typed AST before Z3.
    """

    declaration = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^e[0-9]+$"},
            "mentions": {
                "type": "array", "minItems": 1, "maxItems": 16,
                "items": {"type": "string", "minLength": 1, "maxLength": 256},
            },
        },
        "required": ["id", "mentions"],
        "additionalProperties": False,
    }
    predicate_declaration = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^p[0-9]+$"},
            "mentions": declaration["properties"]["mentions"],
            "arity": {"type": "integer", "minimum": 1, "maximum": MAX_TERMS_PER_ATOM},
        },
        "required": ["id", "mentions", "arity"],
        "additionalProperties": False,
    }
    prefix = {
        "type": "array",
        "minItems": 1,
        "maxItems": MAX_PREFIX_TOKENS,
        "items": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_IDENTIFIER_LENGTH,
        },
    }
    return {
        "type": "object",
        "properties": {
            "translation": {
                "type": "object",
                "properties": {
                    "entities": {"type": "array", "maxItems": 128, "items": declaration},
                    "predicates": {
                        "type": "array", "maxItems": 128, "items": predicate_declaration
                    },
                    "premises": {
                        "type": "array",
                        "maxItems": 256,
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_index": {"type": "integer", "minimum": 0},
                                "prefix": prefix,
                            },
                            "required": ["source_index", "prefix"],
                            "additionalProperties": False,
                        },
                    },
                    "target_prefix": prefix,
                    "options_prefix": {
                        "type": "array", "maxItems": 64, "items": prefix
                    },
                },
                "required": [
                    "entities", "predicates", "premises", "target_prefix", "options_prefix"
                ],
                "additionalProperties": False,
            }
        },
        "required": ["translation"],
        "additionalProperties": False,
    }


def compile_prefix_ir_translation(
    raw: Mapping[str, object],
    *,
    premise_count: int,
    query_type: str,
    option_count: int,
) -> CompiledAstTranslation:
    """Inflate a deterministic prefix stream and compile it as typed FOL."""

    raw = coerce_translation_envelope(raw)
    translation = raw.get("translation") if isinstance(raw, Mapping) else None
    if not isinstance(translation, Mapping):
        raise AstValidationError("missing prefix translation object")
    translation = _infer_prefix_predicate_arities(translation)
    translation = _canonicalize_prefix_symbols(translation)
    raw_predicates = translation.get("predicates", [])
    if not isinstance(raw_predicates, list):
        raise AstValidationError("prefix predicates must be an array")
    arities: Dict[str, int] = {}
    for item in raw_predicates:
        if not isinstance(item, Mapping):
            raise AstValidationError("prefix predicate declaration must be an object")
        predicate = _identifier(item.get("id", ""))
        arity = item.get("arity")
        if not re.fullmatch(r"p\d+", predicate) or predicate in arities:
            raise AstValidationError(f"invalid or duplicate prefix predicate {predicate!r}")
        if not isinstance(arity, int) or not 1 <= arity <= MAX_TERMS_PER_ATOM:
            raise AstValidationError(f"prefix predicate {predicate} has invalid arity")
        arities[predicate] = arity

    premises = translation.get("premises")
    target = translation.get("target_prefix")
    options = translation.get("options_prefix")
    if not isinstance(premises, list) or not isinstance(target, list) or not isinstance(options, list):
        raise AstValidationError("prefix premises, target and options must be arrays")
    nested_premises = []
    for item in premises:
        if not isinstance(item, Mapping):
            raise AstValidationError("prefix premise must be an object")
        nested_premises.append({
            "source_index": item.get("source_index"),
            "ast": _inflate_prefix_formula(item.get("prefix"), arities),
        })
    nested = {
        "translation": {
            "entities": translation.get("entities", []),
            "predicates": raw_predicates,
            "premises": nested_premises,
            "target": _inflate_prefix_formula(target, arities),
            "options": [_inflate_prefix_formula(option, arities) for option in options],
        }
    }
    return compile_ast_translation(
        nested,
        premise_count=premise_count,
        query_type=query_type,
        option_count=option_count,
    )


def _infer_prefix_predicate_arities(
    translation: Mapping[str, object],
) -> Mapping[str, object]:
    """Infer omitted declaration arities from self-delimiting atom tokens.

    Some llama.cpp structured-output backends preserve the object shape but
    omit a nested required integer. ``pN(t1,...,tk)`` already carries exactly
    the same type information, so this is deterministic type inference, not a
    semantic repair.
    """

    normalized = copy.deepcopy(dict(translation))
    observed: Dict[str, Set[int]] = {}

    def scan(tokens: object) -> None:
        if not isinstance(tokens, list):
            return
        for raw_token in tokens:
            if not isinstance(raw_token, str):
                continue
            match = re.fullmatch(r"\s*(p\d+)\(([^()]*)\)\s*", raw_token, flags=re.I)
            if not match:
                continue
            predicate = match.group(1).lower()
            terms = [term.strip() for term in match.group(2).split(",") if term.strip()]
            observed.setdefault(predicate, set()).add(len(terms))

    premises = normalized.get("premises", [])
    if isinstance(premises, list):
        for premise in premises:
            if isinstance(premise, Mapping):
                scan(premise.get("prefix"))
    scan(normalized.get("target_prefix"))
    options = normalized.get("options_prefix", [])
    if isinstance(options, list):
        for option in options:
            scan(option)

    predicates = normalized.get("predicates", [])
    if not isinstance(predicates, list):
        raise AstValidationError("prefix predicates must be an array")
    repaired = []
    for raw_item in predicates:
        if not isinstance(raw_item, Mapping):
            raise AstValidationError("prefix predicate declaration must be an object")
        item = dict(raw_item)
        predicate = _identifier(item.get("id", ""))
        arities = observed.get(predicate, set())
        if len(arities) > 1:
            raise AstValidationError(
                f"prefix predicate {predicate} occurs with inconsistent arities {sorted(arities)}"
            )
        declared = item.get("arity")
        if not isinstance(declared, int):
            if len(arities) != 1:
                raise AstValidationError(
                    f"prefix predicate {predicate} has no inferable arity"
                )
            item["arity"] = next(iter(arities))
        repaired.append(item)
    normalized["predicates"] = repaired
    return normalized


def _canonicalize_prefix_symbols(
    translation: Mapping[str, object],
) -> Mapping[str, object]:
    """Merge declarations sharing an exact source mention and rewrite tokens."""

    normalized = copy.deepcopy(dict(translation))

    def merge(raw_items: object, prefix: str, with_arity: bool) -> Tuple[List[dict], Dict[str, str]]:
        if not isinstance(raw_items, list):
            raise AstValidationError(f"prefix {prefix} declarations must be an array")
        groups: List[dict] = []
        aliases: Dict[str, str] = {}
        mention_owner: Dict[object, str] = {}
        by_id: Dict[str, dict] = {}
        parent: Dict[str, str] = {}

        def find(item_id: str) -> str:
            parent.setdefault(item_id, item_id)
            if parent[item_id] != item_id:
                parent[item_id] = find(parent[item_id])
            return parent[item_id]

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            canonical = min((left_root, right_root), key=lambda value: int(value[1:]))
            parent[right_root if canonical == left_root else left_root] = canonical

        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                raise AstValidationError(f"prefix {prefix} declaration must be an object")
            item = dict(raw_item)
            item_id = _identifier(item.get("id", ""))
            if not re.fullmatch(fr"{prefix}\d+", item_id) or item_id in by_id:
                raise AstValidationError(f"invalid or duplicate prefix symbol {item_id!r}")
            mentions = item.get("mentions")
            if not isinstance(mentions, list) or not mentions:
                raise AstValidationError(f"prefix symbol {item_id} requires mentions")
            item["id"] = item_id
            by_id[item_id] = item
            find(item_id)
            for mention in mentions:
                normalized_mention = _normalize_mention(mention)
                key = (
                    (normalized_mention, item.get("arity"))
                    if with_arity else normalized_mention
                )
                owner = mention_owner.get(key)
                if owner is None:
                    mention_owner[key] = item_id
                else:
                    union(item_id, owner)

        grouped: Dict[str, dict] = {}
        for item_id, item in by_id.items():
            root = find(item_id)
            aliases[item_id] = root
            group = grouped.setdefault(root, {"id": root, "mentions": []})
            for mention in item["mentions"]:
                if mention not in group["mentions"]:
                    group["mentions"].append(mention)
            if with_arity:
                arity = item.get("arity")
                existing = group.get("arity")
                if existing is not None and existing != arity:
                    raise AstValidationError(
                        f"cannot merge prefix predicate aliases with different arities: {root}"
                    )
                group["arity"] = arity
        groups.extend(sorted(grouped.values(), key=lambda item: int(item["id"][1:])))
        return groups, aliases

    entities, entity_aliases = merge(normalized.get("entities", []), "e", False)
    predicates, predicate_aliases = merge(normalized.get("predicates", []), "p", True)
    aliases = {**entity_aliases, **predicate_aliases}

    def rewrite(tokens: object) -> object:
        if not isinstance(tokens, list):
            return tokens
        return [aliases.get(str(token).strip().lower(), token) for token in tokens]

    normalized["entities"] = entities
    normalized["predicates"] = predicates
    premises = normalized.get("premises", [])
    if isinstance(premises, list):
        for premise in premises:
            if isinstance(premise, dict):
                premise["prefix"] = rewrite(premise.get("prefix"))
    normalized["target_prefix"] = rewrite(normalized.get("target_prefix"))
    options = normalized.get("options_prefix", [])
    if isinstance(options, list):
        normalized["options_prefix"] = [rewrite(option) for option in options]
    return normalized


def _inflate_prefix_formula(
    raw_tokens: object, predicate_arities: Mapping[str, int]
) -> Mapping[str, object]:
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise AstValidationError("prefix formula requires a non-empty token array")
    if len(raw_tokens) > MAX_PREFIX_TOKENS:
        raise AstValidationError(f"prefix formula exceeds {MAX_PREFIX_TOKENS} tokens")
    if any(not isinstance(token, str) or not token.strip() for token in raw_tokens):
        raise AstValidationError("prefix tokens must be non-empty strings")
    tokens = [token.strip().lower() for token in raw_tokens]
    cursor = 0

    def take(label: str) -> str:
        nonlocal cursor
        if cursor >= len(tokens):
            raise AstValidationError(f"prefix formula ended while reading {label}")
        token = tokens[cursor]
        cursor += 1
        return token

    def parse(depth: int = 0) -> Mapping[str, object]:
        if depth > MAX_AST_DEPTH:
            raise AstValidationError(f"prefix AST exceeds depth {MAX_AST_DEPTH}")
        op = take("operator")
        compact_quantifier = re.fullmatch(r"(forall|exists):([a-z_][a-z0-9_]*)", op)
        if compact_quantifier:
            quantifier, variable = compact_quantifier.groups()
            return {
                "op": quantifier,
                "vars": [variable],
                "body": parse(depth + 1),
            }
        compact_atom = re.fullmatch(r"(p\d+)\(([^()]*)\)", op)
        if compact_atom:
            predicate, raw_terms = compact_atom.groups()
            arity = predicate_arities.get(predicate)
            if arity is None:
                raise AstValidationError(f"prefix atom uses undeclared predicate {predicate!r}")
            terms = [term.strip() for term in raw_terms.split(",") if term.strip()]
            if len(terms) != arity:
                raise AstValidationError(
                    f"prefix atom {predicate} expects {arity} terms, got {len(terms)}"
                )
            if any(not re.fullmatch(r"[a-z_][a-z0-9_]*", term) for term in terms):
                raise AstValidationError(f"prefix atom {predicate} has invalid terms")
            return {"op": "atom", "predicate": predicate, "terms": terms}
        if op == "atom":
            predicate = take("atom predicate")
            arity = predicate_arities.get(predicate)
            if arity is None:
                raise AstValidationError(f"prefix atom uses undeclared predicate {predicate!r}")
            terms = [take(f"term {index + 1} of {predicate}") for index in range(arity)]
            return {"op": "atom", "predicate": predicate, "terms": terms}
        if op == "not":
            return {"op": "not", "args": [parse(depth + 1)]}
        if op in {"and", "or"}:
            return {"op": op, "args": [parse(depth + 1), parse(depth + 1)]}
        if op in {"implies", "iff"}:
            return {"op": op, "left": parse(depth + 1), "right": parse(depth + 1)}
        if op in {"forall", "exists"}:
            variable = take(f"{op} variable")
            return {"op": op, "vars": [variable], "body": parse(depth + 1)}
        if op in {"true", "false", "none", "theory_inconsistent"}:
            return {"op": op}
        raise AstValidationError(f"unsupported prefix operator {op!r}")

    formula = parse()
    if cursor != len(tokens):
        raise AstValidationError(
            "prefix formula contains trailing tokens: " + ", ".join(tokens[cursor:cursor + 8])
        )
    return formula


def compile_flat_ast_translation(
    raw: Mapping[str, object],
    *,
    premise_count: int,
    query_type: str,
    option_count: int,
) -> CompiledAstTranslation:
    translation = raw.get("translation") if isinstance(raw, Mapping) else None
    if not isinstance(translation, Mapping):
        raise AstValidationError("missing flat translation object")
    premises = translation.get("premises")
    options = translation.get("options")
    if not isinstance(premises, list) or not isinstance(options, list):
        raise AstValidationError("flat premises and options must be arrays")

    nested_premises = []
    for item in premises:
        if not isinstance(item, Mapping):
            raise AstValidationError("flat premise must be an object")
        nested_premises.append({
            "source_index": item.get("source_index"),
            "ast": _inflate_flat_formula(item.get("formula")),
        })
    nested = {
        "translation": {
            "entities": translation.get("entities", []),
            "predicates": translation.get("predicates", []),
            "premises": nested_premises,
            "target": _inflate_flat_formula(translation.get("target")),
            "options": [_inflate_flat_formula(option) for option in options],
        }
    }
    return compile_ast_translation(
        nested,
        premise_count=premise_count,
        query_type=query_type,
        option_count=option_count,
    )


def compile_compact_fol_translation(
    raw: Mapping[str, object],
    *,
    premise_count: int,
    query_type: str,
    option_count: int,
) -> CompiledAstTranslation:
    """Validate a compact opaque-symbol FOL wire representation.

    This format is substantially cheaper for small local LLMs to emit than a
    JSON node graph. The FOL parser still constructs the actual syntax tree;
    no raw string reaches Z3 without syntax, scope, symbol, and arity checks.
    """
    raw = coerce_translation_envelope(raw)
    translation = raw.get("translation") if isinstance(raw, Mapping) else None
    if not isinstance(translation, Mapping):
        raise AstValidationError("missing compact translation object")
    translation = _canonicalize_compact_symbols(translation)

    raw_entities = translation.get("entities", [])
    if not isinstance(raw_entities, list):
        raise AstValidationError("compact entities must be an array")
    entity_ids: Set[str] = set()
    seen_entity_mentions: Dict[str, str] = {}
    for item in raw_entities:
        if not isinstance(item, Mapping):
            raise AstValidationError("compact entity declaration must be an object")
        entity_id = _identifier(item.get("id", ""))
        if not re.fullmatch(r"e\d+", entity_id) or entity_id in entity_ids:
            raise AstValidationError(f"invalid or duplicate compact entity id {entity_id!r}")
        mentions = item.get("mentions")
        if not isinstance(mentions, list) or not mentions:
            raise AstValidationError(f"entity {entity_id} requires source mentions")
        entity_ids.add(entity_id)
        for mention in mentions:
            normalized = _normalize_mention(mention)
            owner = seen_entity_mentions.get(normalized)
            if owner is not None and owner != entity_id:
                raise AstValidationError(
                    f"entity mention {mention!r} is assigned to both {owner} and {entity_id}"
                )
            seen_entity_mentions[normalized] = entity_id

    raw_predicates = translation.get("predicates", [])
    if not isinstance(raw_predicates, list):
        raise AstValidationError("compact predicates must be an array")
    declared_predicates: Dict[str, int] = {}
    seen_predicate_mentions: Dict[Tuple[str, int], str] = {}
    for item in raw_predicates:
        if not isinstance(item, Mapping):
            raise AstValidationError("compact predicate declaration must be an object")
        predicate_id = _identifier(item.get("id", ""))
        arity = item.get("arity")
        if (
            not re.fullmatch(r"p\d+", predicate_id)
            or predicate_id in declared_predicates
            or not isinstance(arity, int)
            or not 1 <= arity <= MAX_TERMS_PER_ATOM
        ):
            raise AstValidationError(f"invalid compact predicate declaration {predicate_id!r}")
        mentions = item.get("mentions")
        if not isinstance(mentions, list) or not mentions:
            raise AstValidationError(f"predicate {predicate_id} requires source mentions")
        declared_predicates[predicate_id] = arity
        for mention in mentions:
            normalized = (_normalize_mention(mention), arity)
            owner = seen_predicate_mentions.get(normalized)
            if owner is not None and owner != predicate_id:
                raise AstValidationError(
                    f"predicate mention {mention!r} is assigned to both {owner} and {predicate_id}"
                )
            seen_predicate_mentions[normalized] = predicate_id

    raw_premises = translation.get("premises")
    if not isinstance(raw_premises, list):
        raise AstValidationError("compact premises must be an array")
    by_index: Dict[int, str] = {}
    for item in raw_premises:
        if not isinstance(item, Mapping) or not isinstance(item.get("source_index"), int):
            raise AstValidationError("compact premise requires source_index")
        index = int(item["source_index"])
        if index in by_index or not isinstance(item.get("fol"), str):
            raise AstValidationError(f"invalid or duplicate compact premise {index}")
        by_index[index] = item["fol"].strip()
    expected = set(range(premise_count))
    if set(by_index) != expected:
        raise AstValidationError(
            f"compact premise indices must be exactly {sorted(expected)}, got {sorted(by_index)}"
        )

    target = translation.get("target_fol", "")
    options = translation.get("options_fol", [])
    if not isinstance(target, str) or not isinstance(options, list):
        raise AstValidationError("compact target/options have invalid types")
    if query_type == "yes_no_uncertain" and not target.strip():
        raise AstValidationError("compact yes/no query requires target_fol")
    if query_type == "multiple_choice" and target.strip():
        raise AstValidationError("compact multiple-choice target_fol must be empty")
    if query_type == "multiple_choice" and len(options) != option_count:
        raise AstValidationError(f"expected {option_count} compact options, got {len(options)}")
    if query_type != "multiple_choice" and options:
        raise AstValidationError("compact non-multiple-choice translation must not emit options")

    premises = tuple(by_index[index] for index in range(premise_count))
    option_formulas = tuple(str(option).strip() for option in options)
    for formula in (*premises, target.strip(), *option_formulas):
        if formula and formula != "__theory_inconsistent__":
            _validate_compact_formula(
                formula,
                entity_ids=entity_ids,
                declared_predicates=declared_predicates,
            )
    if any(formula == "__theory_inconsistent__" for formula in premises):
        raise AstValidationError("theory_inconsistent is valid only as a complete option")
    if target.strip() == "__theory_inconsistent__":
        raise AstValidationError("theory_inconsistent cannot be a target")

    predicates = tuple(
        f"{name}({', '.join('x' + str(i) for i in range(arity))})"
        for name, arity in sorted(declared_predicates.items())
    )
    return CompiledAstTranslation(
        premises_fol=premises,
        target_fol=target.strip(),
        options_fol=option_formulas,
        predicates=predicates,
        constants=tuple(sorted(entity_ids)),
    )


def _canonicalize_compact_symbols(
    translation: Mapping[str, object],
) -> Mapping[str, object]:
    """Merge duplicate symbol declarations and rewrite formulas canonically."""
    normalized = copy.deepcopy(dict(translation))

    def merge_declarations(
        raw_items: object, prefix: str, *, with_arity: bool
    ) -> Tuple[List[dict], Dict[str, str]]:
        if not isinstance(raw_items, list):
            raise AstValidationError(f"compact {prefix} declarations must be an array")
        parent: Dict[str, str] = {}
        items_by_id: Dict[str, dict] = {}
        mention_owners: Dict[str, str] = {}

        def find(item_id: str) -> str:
            parent.setdefault(item_id, item_id)
            if parent[item_id] != item_id:
                parent[item_id] = find(parent[item_id])
            return parent[item_id]

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            canonical = min(
                (left_root, right_root),
                key=lambda value: int(value[1:]),
            )
            other = right_root if canonical == left_root else left_root
            parent[other] = canonical

        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                raise AstValidationError(f"compact {prefix} declaration must be an object")
            item = dict(raw_item)
            item_id = _identifier(item.get("id", ""))
            if not re.fullmatch(fr"{prefix}\d+", item_id) or item_id in items_by_id:
                raise AstValidationError(f"invalid or duplicate symbol id {item_id!r}")
            mentions = item.get("mentions")
            if not isinstance(mentions, list) or not mentions:
                raise AstValidationError(f"symbol {item_id} requires mentions")
            item["id"] = item_id
            items_by_id[item_id] = item
            find(item_id)
            for mention in mentions:
                key = _normalize_mention(mention)
                owner = mention_owners.get(key)
                if owner is None:
                    mention_owners[key] = item_id
                else:
                    union(item_id, owner)

        grouped: Dict[str, dict] = {}
        alias_map: Dict[str, str] = {}
        for item_id, item in items_by_id.items():
            root = find(item_id)
            alias_map[item_id] = root
            group = grouped.setdefault(root, {"id": root, "mentions": []})
            for mention in item["mentions"]:
                if mention not in group["mentions"]:
                    group["mentions"].append(mention)
            if with_arity:
                arity = item.get("arity")
                existing = group.get("arity")
                if existing is not None and existing != arity:
                    raise AstValidationError(
                        f"cannot merge predicate aliases with different arities: {root}"
                    )
                group["arity"] = arity
        return sorted(grouped.values(), key=lambda item: int(item["id"][1:])), alias_map

    entities, entity_aliases = merge_declarations(
        normalized.get("entities", []), "e", with_arity=False
    )
    predicates, predicate_aliases = merge_declarations(
        normalized.get("predicates", []), "p", with_arity=True
    )
    aliases = {**entity_aliases, **predicate_aliases}

    def rewrite(formula: object) -> object:
        if not isinstance(formula, str) or not aliases:
            return formula
        return re.sub(
            r"\b(?:" + "|".join(map(re.escape, sorted(aliases, key=len, reverse=True))) + r")\b",
            lambda match: aliases[match.group(0)],
            formula,
        )

    normalized["entities"] = entities
    normalized["predicates"] = predicates
    premises = normalized.get("premises", [])
    if isinstance(premises, list):
        for premise in premises:
            if isinstance(premise, dict):
                premise["fol"] = rewrite(premise.get("fol"))
    normalized["target_fol"] = rewrite(normalized.get("target_fol", ""))
    options = normalized.get("options_fol", [])
    if isinstance(options, list):
        normalized["options_fol"] = [rewrite(option) for option in options]
    return normalized


def _validate_compact_formula(
    formula: str,
    *,
    entity_ids: Set[str],
    declared_predicates: Mapping[str, int],
) -> None:
    if len(formula) > 8192:
        raise AstValidationError("compact formula is too long")
    from exact_pipeline.engines.fol_parser import sota_parse_text

    try:
        sota_parse_text(formula)
    except Exception as exc:
        raise AstValidationError(f"compact FOL syntax error: {exc}") from exc

    quantified = set(re.findall(
        r"(?:[∀∃]|\bforall\b|\bexists\b)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(?",
        formula,
    ))
    calls = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)", formula)
    seen_predicates: Set[str] = set()
    for name, raw_terms in calls:
        if name not in declared_predicates:
            continue
        seen_predicates.add(name)
        terms = [term.strip() for term in raw_terms.split(",") if term.strip()]
        if len(terms) != declared_predicates[name]:
            raise AstValidationError(
                f"predicate {name} declared arity {declared_predicates[name]} but used with {len(terms)}"
            )
        for term in terms:
            if term not in quantified and term not in entity_ids:
                raise AstValidationError(f"undeclared compact term {term!r}")
    undeclared_calls = {
        name for name, _ in calls
        if re.fullmatch(r"p\d+", name) and name not in declared_predicates
    }
    if undeclared_calls:
        raise AstValidationError(
            "undeclared compact predicates: " + ", ".join(sorted(undeclared_calls))
        )
    if not seen_predicates:
        raise AstValidationError("compact formula contains no declared predicate")


def _inflate_flat_formula(formula: object) -> Mapping[str, object]:
    if not isinstance(formula, Mapping):
        raise AstValidationError("flat formula must be an object")
    root = formula.get("root")
    nodes = formula.get("nodes")
    if not isinstance(root, int) or not isinstance(nodes, list) or not nodes:
        raise AstValidationError("flat formula requires integer root and non-empty nodes")
    by_id: Dict[int, Mapping[str, object]] = {}
    for node in nodes:
        if not isinstance(node, Mapping) or not isinstance(node.get("id"), int):
            raise AstValidationError("every flat node requires an integer id")
        node_id = int(node["id"])
        if node_id in by_id:
            raise AstValidationError(f"duplicate flat node id {node_id}")
        by_id[node_id] = node
    if root not in by_id:
        raise AstValidationError(f"flat root {root} does not exist")

    visiting: Set[int] = set()
    visited: Set[int] = set()

    def inflate(node_id: int) -> Mapping[str, object]:
        if node_id in visiting:
            raise AstValidationError(f"cycle detected at flat node {node_id}")
        node = by_id.get(node_id)
        if node is None:
            raise AstValidationError(f"flat child node {node_id} does not exist")
        visiting.add(node_id)
        op = str(node.get("op", "")).lower()
        present = set(node)
        allowed_by_op = {
            "atom": {"id", "op", "predicate", "terms"},
            "not": {"id", "op", "children"},
            "and": {"id", "op", "children"},
            "or": {"id", "op", "children"},
            "implies": {"id", "op", "children"},
            "iff": {"id", "op", "children"},
            "forall": {"id", "op", "children", "vars"},
            "exists": {"id", "op", "children", "vars"},
            "true": {"id", "op"},
            "false": {"id", "op"},
            "none": {"id", "op"},
            "theory_inconsistent": {"id", "op"},
        }
        unexpected = present - allowed_by_op.get(op, {"id", "op"})
        if unexpected:
            raise AstValidationError(
                f"flat node {node_id} with op {op!r} has invalid fields: "
                + ", ".join(sorted(unexpected))
            )
        children = node.get("children", [])
        if not isinstance(children, list) or any(not isinstance(child, int) for child in children):
            raise AstValidationError(f"flat node {node_id} children must be integer IDs")
        inflated_children = [inflate(child) for child in children]
        result: Dict[str, object] = {"op": op}
        if op == "atom":
            if children:
                raise AstValidationError(f"atom node {node_id} cannot have children")
            result.update(predicate=node.get("predicate"), terms=node.get("terms"))
        elif op == "not":
            if len(inflated_children) != 1:
                raise AstValidationError(f"not node {node_id} requires one child")
            result["args"] = inflated_children
        elif op in {"and", "or"}:
            if len(inflated_children) < 2:
                raise AstValidationError(f"{op} node {node_id} requires at least two children")
            result["args"] = inflated_children
        elif op in {"implies", "iff"}:
            if len(inflated_children) != 2:
                raise AstValidationError(f"{op} node {node_id} requires two children")
            result.update(left=inflated_children[0], right=inflated_children[1])
        elif op in {"forall", "exists"}:
            if len(inflated_children) != 1:
                raise AstValidationError(f"{op} node {node_id} requires one body child")
            result.update(vars=node.get("vars"), body=inflated_children[0])
        elif op in {"true", "false", "none", "theory_inconsistent"}:
            if children:
                raise AstValidationError(f"leaf node {node_id} cannot have children")
        else:
            raise AstValidationError(f"unsupported flat AST op {op!r}")
        visiting.remove(node_id)
        visited.add(node_id)
        return result

    inflated = inflate(root)
    unreachable = set(by_id) - visited
    if unreachable:
        raise AstValidationError(
            "flat formula contains unreachable nodes: " + ", ".join(map(str, sorted(unreachable)))
        )
    return inflated


def compile_ast_translation(
    raw: Mapping[str, object],
    *,
    premise_count: int,
    query_type: str,
    option_count: int,
) -> CompiledAstTranslation:
    translation = raw.get("translation") if isinstance(raw, Mapping) else None
    if not isinstance(translation, Mapping):
        raise AstValidationError("missing translation object")

    entity_ids: Set[str] = set()
    seen_mentions: Dict[str, str] = {}
    registry_present = "entities" in translation
    raw_entities = translation.get("entities", [])
    if not isinstance(raw_entities, list):
        raise AstValidationError("entities must be an array")
    for item in raw_entities:
        if not isinstance(item, Mapping):
            raise AstValidationError("each entity declaration must be an object")
        entity_id = _identifier(item.get("id", ""))
        if not re.fullmatch(r"e\d+", entity_id):
            raise AstValidationError(f"entity id must use opaque e<number> form, got {entity_id!r}")
        if entity_id in entity_ids:
            raise AstValidationError(f"duplicate entity id {entity_id}")
        mentions = item.get("mentions")
        if not isinstance(mentions, list) or not mentions:
            raise AstValidationError(f"entity {entity_id} requires at least one source mention")
        entity_ids.add(entity_id)
        for mention in mentions:
            normalized = _normalize_mention(mention)
            owner = seen_mentions.get(normalized)
            if owner is not None and owner != entity_id:
                raise AstValidationError(
                    f"source mention {mention!r} is assigned to both {owner} and {entity_id}"
                )
            seen_mentions[normalized] = entity_id

    predicate_registry_present = "predicates" in translation
    raw_predicates = translation.get("predicates", [])
    if not isinstance(raw_predicates, list):
        raise AstValidationError("predicates must be an array")
    declared_predicates: Dict[str, int] = {}
    seen_predicate_mentions: Dict[Tuple[str, int], str] = {}
    for item in raw_predicates:
        if not isinstance(item, Mapping):
            raise AstValidationError("each predicate declaration must be an object")
        predicate_id = _identifier(item.get("id", ""))
        if not re.fullmatch(r"p\d+", predicate_id):
            raise AstValidationError(
                f"predicate id must use opaque p<number> form, got {predicate_id!r}"
            )
        arity = item.get("arity")
        if not isinstance(arity, int) or not 1 <= arity <= MAX_TERMS_PER_ATOM:
            raise AstValidationError(f"predicate {predicate_id} has invalid arity")
        if predicate_id in declared_predicates:
            raise AstValidationError(f"duplicate predicate id {predicate_id}")
        mentions = item.get("mentions")
        if not isinstance(mentions, list) or not mentions:
            raise AstValidationError(f"predicate {predicate_id} requires source mentions")
        declared_predicates[predicate_id] = arity
        for mention in mentions:
            normalized = (_normalize_mention(mention), arity)
            owner = seen_predicate_mentions.get(normalized)
            if owner is not None and owner != predicate_id:
                raise AstValidationError(
                    f"predicate mention {mention!r} is assigned to both {owner} and {predicate_id}"
                )
            seen_predicate_mentions[normalized] = predicate_id

    raw_premises = translation.get("premises")
    if not isinstance(raw_premises, list):
        raise AstValidationError("premises must be an array")
    by_index: Dict[int, Mapping[str, object]] = {}
    for item in raw_premises:
        if not isinstance(item, Mapping) or not isinstance(item.get("source_index"), int):
            raise AstValidationError("each premise needs an integer source_index")
        index = int(item["source_index"])
        if index in by_index:
            raise AstValidationError(f"duplicate premise source_index {index}")
        ast = item.get("ast")
        if not isinstance(ast, Mapping):
            raise AstValidationError(f"premise {index} has no AST")
        by_index[index] = ast
    expected = set(range(premise_count))
    if set(by_index) != expected:
        raise AstValidationError(
            f"premise indices must be exactly {sorted(expected)}, got {sorted(by_index)}"
        )

    raw_options = translation.get("options")
    if not isinstance(raw_options, list):
        raise AstValidationError("options must be an array")
    if query_type == "multiple_choice" and len(raw_options) != option_count:
        raise AstValidationError(
            f"expected {option_count} option ASTs, got {len(raw_options)}"
        )
    if query_type != "multiple_choice" and raw_options:
        raise AstValidationError("non-multiple-choice translation must not emit option ASTs")

    target = translation.get("target")
    if not isinstance(target, Mapping):
        raise AstValidationError("target must be an AST object")
    if query_type == "yes_no_uncertain" and target.get("op") == "none":
        raise AstValidationError("yes/no query requires a non-empty target AST")
    if query_type == "multiple_choice" and target.get("op") != "none":
        raise AstValidationError("multiple-choice query target must use op=none")
    target_free_terms = _free_term_identifiers(target)
    premise_free_terms = set().union(
        *(_free_term_identifiers(premise) for premise in by_index.values())
    ) if by_index else set()
    option_free_terms = set().union(
        *(_free_term_identifiers(option) for option in raw_options)
    ) if raw_options else set()
    if query_type == "open_ended":
        if target.get("op") == "none" or "answer" not in target_free_terms:
            raise AstValidationError(
                "open-ended query target must contain the reserved free term 'answer'"
            )
        if "answer" in premise_free_terms or "answer" in option_free_terms:
            raise AstValidationError("reserved term 'answer' is valid only in the query target")
    elif "answer" in target_free_terms | premise_free_terms | option_free_terms:
        raise AstValidationError("reserved term 'answer' is valid only for open-ended queries")

    for index, premise in by_index.items():
        if _contains_op(premise, "theory_inconsistent"):
            raise AstValidationError(
                f"premise {index} cannot contain the theory_inconsistent meta operator"
            )
    if _contains_op(target, "theory_inconsistent"):
        raise AstValidationError("target cannot contain the theory_inconsistent meta operator")
    for index, option in enumerate(raw_options):
        if not isinstance(option, Mapping):
            raise AstValidationError(f"option {index} must be an AST object")
        if _contains_op(option, "theory_inconsistent") and option.get("op") != "theory_inconsistent":
            raise AstValidationError(
                f"option {index} may use theory_inconsistent only as its complete top-level meaning"
            )

    # In textbook and generated FOL, free one-letter variables in a rule are
    # conventionally read under universal closure. Elaborate that convention
    # deterministically, while still rejecting undeclared named constants such
    # as ``med_v4`` or ``harbor`` when an entity registry is present.
    def universally_close(node: Mapping[str, object]) -> Mapping[str, object]:
        if node.get("op") in {"none", "theory_inconsistent"}:
            return node
        free_terms = _free_term_identifiers(node) - entity_ids
        implicit_variables = sorted(
            term for term in free_terms
            if re.fullmatch(r"[a-z]", term) or re.fullmatch(r"[xyzuvw][0-9]+", term)
        )
        if not implicit_variables:
            return node
        return {"op": "forall", "vars": implicit_variables, "body": node}

    if registry_present:
        by_index = {index: universally_close(node) for index, node in by_index.items()}
        target = universally_close(target)
        raw_options = [universally_close(option) for option in raw_options]

    context = AstCompileContext()
    premises = tuple(
        compile_ast(by_index[index], context=context, bound=frozenset())
        for index in range(premise_count)
    )
    target_fol = "" if target.get("op") == "none" else compile_ast(
        target, context=context, bound=frozenset()
    )
    options = tuple(
        compile_ast(option, context=context, bound=frozenset())
        for option in raw_options
    )
    if registry_present:
        allowed_reserved = {"answer"} if query_type == "open_ended" else set()
        undeclared = context.constants - entity_ids - allowed_reserved
        if undeclared:
            raise AstValidationError(
                "all constants must use declared opaque entity IDs; undeclared: "
                + ", ".join(sorted(undeclared))
            )
    if predicate_registry_present:
        undeclared_predicates = set(context.predicate_arities) - set(declared_predicates)
        if undeclared_predicates:
            raise AstValidationError(
                "all atoms must use declared opaque predicate IDs; undeclared: "
                + ", ".join(sorted(undeclared_predicates))
            )
        for predicate, actual_arity in context.predicate_arities.items():
            if declared_predicates[predicate] != actual_arity:
                raise AstValidationError(
                    f"predicate {predicate} declared arity {declared_predicates[predicate]} "
                    f"but used with arity {actual_arity}"
                )
    predicates = tuple(
        f"{name}({', '.join('x' + str(i) for i in range(arity))})"
        for name, arity in sorted(context.predicate_arities.items())
    )
    return CompiledAstTranslation(
        premises_fol=premises,
        target_fol=target_fol,
        options_fol=options,
        predicates=predicates,
        constants=tuple(sorted(context.constants - {"answer"})),
        entity_mentions=tuple(
            (
                _identifier(item.get("id", "")),
                tuple(str(mention) for mention in item.get("mentions", [])),
            )
            for item in raw_entities
            if isinstance(item, Mapping)
        ),
        predicate_mentions=tuple(
            (
                _identifier(item.get("id", "")),
                tuple(str(mention) for mention in item.get("mentions", [])),
            )
            for item in raw_predicates
            if isinstance(item, Mapping)
        ),
    )


def _contains_op(node: object, desired: str) -> bool:
    if isinstance(node, Mapping):
        if node.get("op") == desired:
            return True
        return any(_contains_op(value, desired) for value in node.values())
    if isinstance(node, list):
        return any(_contains_op(value, desired) for value in node)
    return False


def compile_ast(
    node: Mapping[str, object],
    *,
    context: AstCompileContext,
    bound: frozenset[str],
    depth: int = 0,
) -> str:
    if not isinstance(node, Mapping):
        raise AstValidationError("AST node must be an object")
    context.nodes_seen += 1
    if context.nodes_seen > MAX_AST_NODES:
        raise AstValidationError(f"AST exceeds {MAX_AST_NODES} nodes")
    if depth > MAX_AST_DEPTH:
        raise AstValidationError(f"AST exceeds depth {MAX_AST_DEPTH}")
    op = str(node.get("op", "")).strip().lower()
    if op not in _OPS:
        raise AstValidationError(f"unsupported AST op {op!r}")
    if op == "none":
        raise AstValidationError("op=none is only valid as the top-level absent target")
    if op == "true":
        return "(__logic_truth__ = __logic_truth__)"
    if op == "false":
        return "¬(__logic_truth__ = __logic_truth__)"
    if op == "theory_inconsistent":
        return "__theory_inconsistent__"
    if op == "atom":
        return _compile_atom(node, context=context, bound=bound)
    if op == "not":
        args = node.get("args")
        child = args[0] if isinstance(args, list) and len(args) == 1 else node.get("body")
        if not isinstance(child, Mapping):
            raise AstValidationError("not requires one child")
        return f"¬{_parenthesize(compile_ast(child, context=context, bound=bound, depth=depth + 1))}"
    if op in {"and", "or"}:
        args = node.get("args")
        if not isinstance(args, list) or len(args) < 2:
            raise AstValidationError(f"{op} requires at least two children")
        if len(args) > MAX_CONNECTIVE_CHILDREN:
            raise AstValidationError(
                f"{op} has {len(args)} children; maximum is {MAX_CONNECTIVE_CHILDREN}"
            )
        symbol = " ∧ " if op == "and" else " ∨ "
        return "(" + symbol.join(
            compile_ast(child, context=context, bound=bound, depth=depth + 1)
            for child in _mapping_children(args, op)
        ) + ")"
    if op in {"implies", "iff"}:
        left, right = node.get("left"), node.get("right")
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            args = node.get("args")
            if isinstance(args, list) and len(args) == 2:
                left, right = args
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise AstValidationError(f"{op} requires left and right children")
        symbol = " → " if op == "implies" else " ↔ "
        return "(" + compile_ast(left, context=context, bound=bound, depth=depth + 1) + symbol + compile_ast(
            right, context=context, bound=bound, depth=depth + 1
        ) + ")"
    if op in {"forall", "exists"}:
        variables = node.get("vars")
        body = node.get("body")
        if not isinstance(variables, list) or not variables or not isinstance(body, Mapping):
            raise AstValidationError(f"{op} requires vars and body")
        if len(variables) > MAX_VARS_PER_QUANTIFIER:
            raise AstValidationError(
                f"{op} has {len(variables)} variables; maximum is {MAX_VARS_PER_QUANTIFIER}"
            )
        normalized_vars = tuple(_identifier(variable) for variable in variables)
        if len(set(normalized_vars)) != len(normalized_vars):
            raise AstValidationError(f"{op} contains duplicate variables")
        free_in_body = _free_term_identifiers(body)
        unused = [variable for variable in normalized_vars if variable not in free_in_body]
        if unused:
            raise AstValidationError(
                f"{op} contains variables unused in its body: {', '.join(unused)}"
            )
        nested_bound = frozenset(set(bound) | set(normalized_vars))
        compiled = compile_ast(body, context=context, bound=nested_bound, depth=depth + 1)
        quantifier = "∀" if op == "forall" else "∃"
        for variable in reversed(normalized_vars):
            compiled = f"{quantifier}{variable}({compiled})"
        return compiled
    raise AstValidationError(f"unhandled AST op {op}")


def _compile_atom(
    node: Mapping[str, object],
    *,
    context: AstCompileContext,
    bound: frozenset[str],
) -> str:
    predicate_raw = node.get("predicate")
    terms = node.get("terms")
    if not isinstance(predicate_raw, str) or not isinstance(terms, list) or not terms:
        raise AstValidationError("atom requires predicate and at least one term")
    if len(terms) > MAX_TERMS_PER_ATOM:
        raise AstValidationError(
            f"atom has {len(terms)} terms; maximum is {MAX_TERMS_PER_ATOM}"
        )
    predicate, lexically_negated = _canonical_predicate(predicate_raw)
    normalized_terms: List[str] = []
    for raw_term in terms:
        if not isinstance(raw_term, str) or not raw_term.strip():
            raise AstValidationError("atom terms must be non-empty strings")
        term = _identifier(raw_term)
        if term not in bound:
            context.constants.add(term)
        normalized_terms.append(term)
    arity = len(normalized_terms)
    existing = context.predicate_arities.get(predicate)
    if existing is not None and existing != arity:
        raise AstValidationError(
            f"predicate {predicate} has inconsistent arity {existing} vs {arity}"
        )
    context.predicate_arities[predicate] = arity
    atom = f"{predicate}({', '.join(normalized_terms)})"
    return f"¬{atom}" if lexically_negated else atom


def _canonical_predicate(raw: str) -> Tuple[str, bool]:
    name = _identifier(raw)
    negated = False
    prefixes = ("does_not_", "do_not_", "did_not_", "cannot_", "can_not_", "not_")
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            negated = True
            break
    if name.startswith("lacks_"):
        name = "has_" + name[len("lacks_"):]
        negated = True
    if name.startswith("fails_"):
        name = "passes_" + name[len("fails_"):]
        negated = True
    if name.startswith("have_"):
        name = "has_" + name[len("have_"):]
    if not name:
        raise AstValidationError("predicate becomes empty after polarity normalization")
    return name, negated


def _identifier(raw: object) -> str:
    raw_text = str(raw).strip()
    if len(raw_text) > MAX_IDENTIFIER_LENGTH:
        raise AstValidationError(
            f"identifier exceeds {MAX_IDENTIFIER_LENGTH} characters"
        )
    text = re.sub(r"[^A-Za-z0-9_]+", "_", raw_text).strip("_").lower()
    text = re.sub(r"_+", "_", text)
    if not text:
        raise AstValidationError(f"invalid identifier {raw!r}")
    if text[0].isdigit():
        text = "symbol_" + text
    return text


def _normalize_mention(raw: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(raw).casefold()).strip()
    if not text:
        raise AstValidationError("entity mentions must contain letters or digits")
    return " ".join(text.split())


def _mapping_children(args: Sequence[object], op: str) -> Iterable[Mapping[str, object]]:
    for child in args:
        if not isinstance(child, Mapping):
            raise AstValidationError(f"{op} children must be AST objects")
        yield child


def _free_term_identifiers(node: object, inner_bound: frozenset[str] = frozenset()) -> Set[str]:
    """Return term identifiers not captured by quantifiers inside ``node``."""
    if not isinstance(node, Mapping):
        return set()
    op = str(node.get("op", "")).strip().lower()
    if op == "atom":
        terms = node.get("terms")
        if not isinstance(terms, list):
            return set()
        return {
            _identifier(term)
            for term in terms
            if _identifier(term) not in inner_bound
        }
    if op in {"forall", "exists"}:
        variables = node.get("vars")
        body = node.get("body")
        locally_bound = {
            _identifier(variable) for variable in variables
        } if isinstance(variables, list) else set()
        return _free_term_identifiers(
            body, frozenset(set(inner_bound) | locally_bound)
        )
    free: Set[str] = set()
    for key in ("args", "left", "right", "body"):
        value = node.get(key)
        if isinstance(value, list):
            for child in value:
                free.update(_free_term_identifiers(child, inner_bound))
        else:
            free.update(_free_term_identifiers(value, inner_bound))
    return free


def _parenthesize(formula: str) -> str:
    if formula.startswith("(") and formula.endswith(")"):
        return formula
    return f"({formula})"
