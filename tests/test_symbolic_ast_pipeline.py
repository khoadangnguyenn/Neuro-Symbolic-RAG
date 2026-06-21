import unittest
import time
import threading
from collections import OrderedDict
from types import SimpleNamespace

from exact_pipeline.engines.fol_ast import (
    AstValidationError,
    compile_ast_translation,
    compile_compact_fol_translation,
    compile_flat_ast_translation,
    compile_prefix_ir_translation,
    logic_ast_json_schema,
    logic_flat_ast_json_schema,
    logic_prefix_ir_json_schema,
)
from exact_pipeline.engines.symbolic_solver import run_symbolic_solver
from exact_pipeline.engines.logic import LogicPipeline
from exact_pipeline.llm.llm import LLMError


def atom(predicate, *terms):
    return {"op": "atom", "predicate": predicate, "terms": list(terms)}


def solve(raw, premise_count, query_type, option_count, intent="choose_true"):
    compiled = compile_ast_translation(
        raw,
        premise_count=premise_count,
        query_type=query_type,
        option_count=option_count,
    )
    return run_symbolic_solver(
        {"query_type": query_type, "intent": intent},
        {
            "predicates": list(compiled.predicates),
            "functions": [],
            "premises_fol": list(compiled.premises_fol),
            "condition_fol": "",
            "target_fol": compiled.target_fol,
            "options_fol": list(compiled.options_fol),
        },
    )


class SymbolicAstPipelineTest(unittest.TestCase):
    def test_inconsistent_supplied_fol_is_preflighted_without_llm_retranslation(self):
        class LlmMustNotRun:
            enabled = True

            def chat_json(self, **kwargs):
                raise AssertionError("inconsistent trusted FOL must be decided before semantic parsing")

        pipeline = object.__new__(LogicPipeline)
        pipeline.llm = LlmMustNotRun()
        pipeline.max_retries = 2
        result = pipeline._answer_with_symbolic_ast(
            question="Does Asha have access to a secure facility?",
            premises_nl=[],
            premises_fol=[
                "∀x(researcher(x)→∃y(lab_access(x,y)∧secure_facility(y)))",
                "∀x∀y((researcher(x)∧lab_access(x,y)∧secure_facility(y))→has_key(x))",
                "researcher(asha)",
                "¬has_key(asha)",
            ],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
            deadline=time.monotonic() + 55,
        )
        self.assertEqual(result.answer, "Uncertain")
        self.assertEqual(result.source, "symbolic-fol-preflight")
        self.assertEqual(result.metadata["logical_status"], "Inconsistent")

    def test_inconsistent_yes_no_is_normalized_to_uncertain_with_status(self):
        class FakeParser:
            enabled = True

            def chat_json(self, **kwargs):
                return {
                    "entities": [{"id": "e0", "mentions": ["Asha"]}],
                    "predicates": [{"id": "p0", "mentions": ["has key"], "arity": 1}],
                    "premises": [
                        {"source_index": 0, "fol": "p0(e0)"},
                        {"source_index": 1, "fol": "¬p0(e0)"},
                    ],
                    "target_fol": "p0(e0)",
                    "options_fol": [],
                }

        pipeline = object.__new__(LogicPipeline)
        pipeline.llm = FakeParser()
        pipeline.max_retries = 1
        result = pipeline._answer_with_symbolic_ast(
            question="Does Asha have a key?",
            premises_nl=["Asha has a key.", "Asha does not have a key."],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
            deadline=time.monotonic() + 55,
        )
        self.assertEqual(result.answer, "Uncertain")
        self.assertEqual(result.metadata["logical_status"], "Inconsistent")
        self.assertEqual(
            result.metadata["normalization_policy"], "inconsistent_to_uncertain"
        )

    def test_compact_fol_wire_is_short_validated_and_solver_ready(self):
        raw = {"translation": {
            "entities": [{"id": "e0", "mentions": ["Vehicle Delta"]}],
            "predicates": [
                {"id": "p0", "mentions": ["passes test"], "arity": 1},
                {"id": "p1", "mentions": ["has permit"], "arity": 1},
                {"id": "p2", "mentions": ["has LiDAR"], "arity": 1},
            ],
            "premises": [
                {"source_index": 0, "fol": "∀x(p0(x)→p1(x))"},
                {"source_index": 1, "fol": "∀x(p1(x)→p2(x))"},
                {"source_index": 2, "fol": "¬p2(e0)"},
            ],
            "target_fol": "p0(e0)",
            "options_fol": [],
        }}
        compiled = compile_compact_fol_translation(
            raw, premise_count=3, query_type="yes_no_uncertain", option_count=0
        )
        result = run_symbolic_solver(
            {"query_type": "yes_no_uncertain", "intent": "verify_true"},
            {
                "predicates": list(compiled.predicates),
                "functions": [],
                "premises_fol": list(compiled.premises_fol),
                "target_fol": compiled.target_fol,
                "options_fol": [],
            },
        )
        self.assertEqual(result["verdict"], "False")

    def test_prefix_ir_handles_biconditional_chain_without_fol_strings(self):
        raw = {"translation": {
            "entities": [{"id": "e0", "mentions": ["Greenhouse Basil"]}],
            "predicates": [
                {"id": "p0", "mentions": ["triggers autonomous watering"], "arity": 1},
                {"id": "p1", "mentions": ["soil moisture below 30 percent"], "arity": 1},
                {"id": "p2", "mentions": ["heatwave sensor active"], "arity": 1},
            ],
            "premises": [
                {"source_index": 0, "prefix": [
                    "forall:x", "iff", "p0(x)", "p1(x)"
                ]},
                {"source_index": 1, "prefix": [
                    "forall:x", "implies", "p2(x)", "p1(x)"
                ]},
                {"source_index": 2, "prefix": ["p2(e0)"]},
            ],
            "target_prefix": ["p0(e0)"],
            "options_prefix": [],
        }}
        compiled = compile_prefix_ir_translation(
            raw, premise_count=3, query_type="yes_no_uncertain", option_count=0
        )
        result = run_symbolic_solver(
            {"query_type": "yes_no_uncertain", "intent": "verify_true"},
            {
                "predicates": list(compiled.predicates),
                "functions": [],
                "premises_fol": list(compiled.premises_fol),
                "target_fol": compiled.target_fol,
                "options_fol": [],
            },
        )
        self.assertEqual(result["verdict"], "True")

    def test_prefix_ir_preserves_nested_existence_and_unsat_meta_option(self):
        raw = {"translation": {
            "entities": [{"id": "e0", "mentions": ["Harbor Microgrid"]}],
            "predicates": [
                {"id": "p0", "mentions": ["secure microgrid"], "arity": 1},
                {"id": "p1", "mentions": ["backup generator belongs to microgrid"], "arity": 2},
                {"id": "p2", "mentions": ["passes weekly load test"], "arity": 1},
                {"id": "p3", "mentions": ["needs a generator"], "arity": 1},
            ],
            "premises": [
                {"source_index": 0, "prefix": [
                    "forall", "x", "implies", "atom", "p0", "x",
                    "exists", "g", "atom", "p1", "g", "x"
                ]},
                {"source_index": 1, "prefix": [
                    "forall", "x", "forall", "g", "implies",
                    "and", "atom", "p0", "x", "atom", "p1", "g", "x",
                    "atom", "p2", "g"
                ]},
                {"source_index": 2, "prefix": [
                    "not", "exists", "g", "atom", "p1", "g", "e0"
                ]},
                {"source_index": 3, "prefix": ["atom", "p0", "e0"]},
            ],
            "target_prefix": ["none"],
            "options_prefix": [
                ["forall", "g", "implies", "atom", "p1", "g", "e0", "atom", "p2", "g"],
                ["theory_inconsistent"],
                ["atom", "p3", "e0"],
            ],
        }}
        compiled = compile_prefix_ir_translation(
            raw, premise_count=4, query_type="multiple_choice", option_count=3
        )
        result = run_symbolic_solver(
            {"query_type": "multiple_choice", "intent": "choose_true"},
            {
                "predicates": list(compiled.predicates),
                "functions": [],
                "premises_fol": list(compiled.premises_fol),
                "target_fol": "",
                "options_fol": list(compiled.options_fol),
            },
        )
        self.assertEqual(result["best_option"], "B")

    def test_prefix_ir_rejects_missing_quantifier_body_and_trailing_tokens(self):
        base = {"translation": {
            "entities": [],
            "predicates": [{"id": "p0", "mentions": ["ready"], "arity": 1}],
            "premises": [{"source_index": 0, "prefix": ["forall", "x"]}],
            "target_prefix": ["atom", "p0", "x"],
            "options_prefix": [],
        }}
        with self.assertRaisesRegex(AstValidationError, "ended while reading operator"):
            compile_prefix_ir_translation(
                base, premise_count=1, query_type="yes_no_uncertain", option_count=0
            )
        base["translation"]["premises"][0]["prefix"] = [
            "atom", "p0", "x", "atom", "p0", "x"
        ]
        with self.assertRaisesRegex(AstValidationError, "trailing tokens"):
            compile_prefix_ir_translation(
                base, premise_count=1, query_type="yes_no_uncertain", option_count=0
            )

    def test_prefix_ir_merges_duplicate_mentions_and_rewrites_all_formula_tokens(self):
        raw = {"translation": {
            "entities": [
                {"id": "e2", "mentions": ["Harbor Microgrid"]},
                {"id": "e4", "mentions": ["Harbor Microgrid"]},
            ],
            "predicates": [
                {"id": "p1", "mentions": ["is secure"], "arity": 1},
                {"id": "p3", "mentions": ["is secure"], "arity": 1},
            ],
            "premises": [{"source_index": 0, "prefix": ["atom", "p3", "e4"]}],
            "target_prefix": ["atom", "p1", "e2"],
            "options_prefix": [],
        }}
        compiled = compile_prefix_ir_translation(
            raw, premise_count=1, query_type="yes_no_uncertain", option_count=0
        )
        self.assertEqual(compiled.premises_fol, ("p1(e2)",))
        self.assertEqual(compiled.target_fol, "p1(e2)")

    def test_prefix_ir_infers_omitted_arity_from_atom_tokens(self):
        raw = {"translation": {
            "entities": [{"id": "e0", "mentions": ["Omega"]}],
            "predicates": [
                {"id": "p0", "mentions": ["ready"]},
                {"id": "p1", "mentions": ["connected"]},
            ],
            "premises": [
                {"source_index": 0, "prefix": ["p0(e0)"]},
                {"source_index": 1, "prefix": ["exists:x", "p1(e0,x)"]},
            ],
            "target_prefix": ["p0(e0)"],
            "options_prefix": [],
        }}
        compiled = compile_prefix_ir_translation(
            raw, premise_count=2, query_type="yes_no_uncertain", option_count=0
        )
        self.assertIn("p0(x0)", compiled.predicates)
        self.assertIn("p1(x0, x1)", compiled.predicates)

    def test_prefix_ir_universally_closes_free_rule_variables(self):
        raw = {"translation": {
            "entities": [{"id": "e0", "mentions": ["Alpha"]}],
            "predicates": [
                {"id": "p0", "mentions": ["ready"], "arity": 1},
                {"id": "p1", "mentions": ["approved"], "arity": 1},
            ],
            "premises": [
                {"source_index": 0, "prefix": ["implies", "p0(x)", "p1(x)"]},
                {"source_index": 1, "prefix": ["p0(e0)"]},
            ],
            "target_prefix": ["p1(e0)"],
            "options_prefix": [],
        }}
        compiled = compile_prefix_ir_translation(
            raw, premise_count=2, query_type="yes_no_uncertain", option_count=0
        )
        self.assertTrue(compiled.premises_fol[0].startswith("∀x("))
        result = run_symbolic_solver(
            {"query_type": "yes_no_uncertain", "intent": "verify_true"},
            {
                "predicates": list(compiled.predicates),
                "functions": [],
                "premises_fol": list(compiled.premises_fol),
                "target_fol": compiled.target_fol,
                "options_fol": [],
            },
        )
        self.assertEqual(result["verdict"], "True")

    def test_same_surface_predicate_with_different_arity_is_typed_not_merged(self):
        raw = {"translation": {
            "entities": [{"id": "e0", "mentions": ["Harbor"]}],
            "predicates": [
                {"id": "p0", "mentions": ["backup generator"], "arity": 1},
                {"id": "p1", "mentions": ["backup generator"], "arity": 2},
            ],
            "premises": [
                {"source_index": 0, "prefix": ["p0(e0)"]},
                {"source_index": 1, "prefix": ["p1(e0,e0)"]},
            ],
            "target_prefix": ["p0(e0)"],
            "options_prefix": [],
        }}
        compiled = compile_prefix_ir_translation(
            raw, premise_count=2, query_type="yes_no_uncertain", option_count=0
        )
        self.assertIn("p0(x0)", compiled.predicates)
        self.assertIn("p1(x0, x1)", compiled.predicates)

    def test_open_numeric_query_projects_only_entailed_value(self):
        raw = {"translation": {
            "entities": [
                {"id": "e0", "mentions": ["Model Vision-B"]},
                {"id": "e1", "mentions": ["84 percent"]},
                {"id": "e2", "mentions": ["45 milliseconds"]},
                {"id": "e3", "mentions": ["12 milliseconds"]},
            ],
            "predicates": [
                {"id": "p0", "mentions": ["trained on ImageNet-X"], "arity": 1},
                {"id": "p1", "mentions": ["baseline accuracy"], "arity": 2},
                {"id": "p2", "mentions": ["inference latency"], "arity": 2},
                {"id": "p3", "mentions": ["undergoes FP16 quantization"], "arity": 1},
            ],
            "premises": [
                {"source_index": 0, "prefix": [
                    "forall:x", "implies", "p0(x)", "p1(x,e1)"
                ]},
                {"source_index": 1, "prefix": [
                    "forall:x", "implies", "p1(x,e1)", "p2(x,e2)"
                ]},
                {"source_index": 2, "prefix": [
                    "forall:x", "implies", "and", "p2(x,e2)", "p3(x)", "p2(x,e3)"
                ]},
                {"source_index": 3, "prefix": ["p0(e0)"]},
                {"source_index": 4, "prefix": ["not", "p3(e0)"]},
            ],
            "target_prefix": ["p2(e0,answer)"],
            "options_prefix": [],
        }}
        compiled = compile_prefix_ir_translation(
            raw, premise_count=5, query_type="open_ended", option_count=0
        )
        result = run_symbolic_solver(
            {"query_type": "open_ended", "intent": "open_analysis"},
            {
                "predicates": list(compiled.predicates),
                "functions": [],
                "premises_fol": list(compiled.premises_fol),
                "target_fol": compiled.target_fol,
                "options_fol": [],
                "constant_labels": {
                    entity_id: mentions[0]
                    for entity_id, mentions in compiled.entity_mentions
                },
            },
        )
        self.assertEqual(result["verdict"], "Answer")
        self.assertEqual(result["answer"], "45")
        self.assertEqual(result["unit"], "milliseconds")
        self.assertEqual(set(result["premises_used"]), {"P1", "P2", "P4"})

    def test_open_entity_query_projects_derived_named_entity(self):
        raw = {"translation": {
            "entities": [
                {"id": "e0", "mentions": ["Alice"]},
                {"id": "e1", "mentions": ["Bob"]},
                {"id": "e2", "mentions": ["Charlie"]},
            ],
            "predicates": [
                {"id": "p0", "mentions": ["architected microservices"], "arity": 1},
                {"id": "p1", "mentions": ["masters gRPC"], "arity": 1},
                {"id": "p2", "mentions": ["security clearance"], "arity": 1},
                {"id": "p3", "mentions": ["assigned to Project Titan"], "arity": 1},
            ],
            "premises": [
                {"source_index": 0, "prefix": [
                    "forall:x", "implies", "p0(x)", "p1(x)"
                ]},
                {"source_index": 1, "prefix": [
                    "forall:x", "implies", "and", "p1(x)", "p2(x)", "p3(x)"
                ]},
                {"source_index": 2, "prefix": ["p1(e0)"]},
                {"source_index": 3, "prefix": ["not", "p2(e0)"]},
                {"source_index": 4, "prefix": ["p0(e1)"]},
                {"source_index": 5, "prefix": ["p2(e1)"]},
                {"source_index": 6, "prefix": ["p2(e2)"]},
                {"source_index": 7, "prefix": ["not", "p0(e2)"]},
            ],
            "target_prefix": ["p3(answer)"],
            "options_prefix": [],
        }}
        compiled = compile_prefix_ir_translation(
            raw, premise_count=8, query_type="open_ended", option_count=0
        )
        result = run_symbolic_solver(
            {"query_type": "open_ended", "intent": "open_analysis"},
            {
                "predicates": list(compiled.predicates),
                "functions": [],
                "premises_fol": list(compiled.premises_fol),
                "target_fol": compiled.target_fol,
                "options_fol": [],
                "constant_labels": {
                    entity_id: mentions[0]
                    for entity_id, mentions in compiled.entity_mentions
                },
            },
        )
        self.assertEqual(result["answer"], "Bob")
        self.assertEqual(set(result["premises_used"]), {"P1", "P2", "P5", "P6"})

    def test_open_query_rejects_semantic_collapse_then_projects_answer(self):
        """A schema-valid but meaning-losing parse must never reach Z3."""

        entities = [
            {"id": "e0", "mentions": ["Model Vision-B"]},
            {"id": "e1", "mentions": ["84 percent"]},
            {"id": "e2", "mentions": ["45 milliseconds"]},
            {"id": "e3", "mentions": ["12 milliseconds"]},
            {"id": "e4", "mentions": ["250 megabytes"]},
        ]
        predicates = [
            {"id": "p0", "mentions": ["trained on ImageNet-X dataset"], "arity": 1},
            {"id": "p1", "mentions": ["baseline accuracy"], "arity": 2},
            {"id": "p2", "mentions": ["inference latency"], "arity": 2},
            {"id": "p3", "mentions": ["undergoes FP16 quantization"], "arity": 1},
            {"id": "p4", "mentions": ["occupies memory"], "arity": 2},
        ]
        bad = {"translation": {
            "entities": entities,
            "predicates": predicates,
            "premises": [
                {"source_index": 0, "prefix": ["forall:x", "implies", "p0(x)", "p1(x,e1)"]},
                # This incorrectly duplicates P1 and drops the latency concept.
                {"source_index": 1, "prefix": ["forall:x", "implies", "p0(x)", "p1(x,e1)"]},
                {"source_index": 2, "prefix": ["forall:x", "implies", "and", "p2(x,e2)", "p3(x)", "p2(x,e3)"]},
                {"source_index": 3, "prefix": ["p0(e0)"]},
                {"source_index": 4, "prefix": ["not", "p3(e0)"]},
                {"source_index": 5, "prefix": ["p4(e0,e4)"]},
            ],
            "target_prefix": ["p2(e0,answer)"],
            "options_prefix": [],
        }}
        good = {"translation": {
            "entities": entities,
            "predicates": predicates,
            "premises": [
                {"source_index": 0, "prefix": ["forall:x", "implies", "p0(x)", "p1(x,e1)"]},
                {"source_index": 1, "prefix": ["forall:x", "implies", "p1(x,e1)", "p2(x,e2)"]},
                {"source_index": 2, "prefix": ["forall:x", "implies", "and", "p2(x,e2)", "p3(x)", "p2(x,e3)"]},
                {"source_index": 3, "prefix": ["p0(e0)"]},
                {"source_index": 4, "prefix": ["not", "p3(e0)"]},
                {"source_index": 5, "prefix": ["p4(e0,e4)"]},
            ],
            "target_prefix": ["p2(e0,answer)"],
            "options_prefix": [],
        }}

        class BadThenGoodParser:
            enabled = True

            def __init__(self):
                self.calls = 0

            def chat_json(self, **kwargs):
                self.calls += 1
                return bad if self.calls == 1 else good

        parser = BadThenGoodParser()
        pipeline = object.__new__(LogicPipeline)
        pipeline.llm = parser
        pipeline.max_retries = 1
        pipeline._symbolic_ast_cache = OrderedDict()
        pipeline._symbolic_ast_cache_lock = threading.RLock()
        pipeline._symbolic_ast_cache_capacity = 8
        result = pipeline._answer_with_symbolic_ast(
            question="What is the inference latency of Model Vision-B in milliseconds?",
            premises_nl=[
                "If a deep learning model is trained on the ImageNet-X dataset, it achieves a baseline accuracy of 84 percent.",
                "Whenever a model achieves a baseline accuracy of 84 percent, its initial inference latency is exactly 45 milliseconds.",
                "If a model's initial inference latency is 45 milliseconds and it successfully undergoes FP16 quantization, its latency drops to 12 milliseconds.",
                "Model Vision-B is trained on the ImageNet-X dataset.",
                "Model Vision-B failed to undergo FP16 quantization.",
                "Model Vision-B occupies 250 megabytes of memory.",
            ],
            premises_fol=[],
            options=[],
            query_type="open_ended",
            intent="open_analysis",
            deadline=time.monotonic() + 55,
        )
        self.assertEqual(parser.calls, 2)
        self.assertEqual(result.answer, "45")
        self.assertEqual(result.unit, "milliseconds")
        self.assertEqual(result.premises_used, [0, 1, 3])
        self.assertTrue(any(
            "semantic alignment" in error
            for error in result.metadata["translation_errors"]
        ))

    def test_compact_repair_recovers_missing_envelope_and_merges_duplicate_entities(self):
        direct_translation = {
            "entities": [
                {"id": "e2", "mentions": ["secure microgrid"]},
                {"id": "e4", "mentions": ["secure microgrid"]},
            ],
            "predicates": [
                {"id": "p0", "mentions": ["is secure"], "arity": 1}
            ],
            "premises": [{"source_index": 0, "fol": "p0(e4)"}],
            "target_fol": "p0(e2)",
            "options_fol": [],
        }
        compiled = compile_compact_fol_translation(
            {"result": {"payload": direct_translation}},
            premise_count=1,
            query_type="yes_no_uncertain",
            option_count=0,
        )
        self.assertEqual(compiled.premises_fol, ("p0(e2)",))
        self.assertEqual(compiled.target_fol, "p0(e2)")
        self.assertEqual(compiled.constants, ("e2",))

    def test_flat_wire_ast_compiles_to_same_typed_logic(self):
        def formula(nodes, root):
            return {"root": root, "nodes": nodes}

        raw = {"translation": {
            "entities": [{"id": "e0", "mentions": ["Vehicle Delta"]}],
            "predicates": [
                {"id": "p0", "mentions": ["passes test"], "arity": 1},
                {"id": "p1", "mentions": ["has permit"], "arity": 1},
                {"id": "p2", "mentions": ["has LiDAR"], "arity": 1},
            ],
            "premises": [
                {"source_index": 0, "formula": formula([
                    {"id": 0, "op": "atom", "predicate": "p0", "terms": ["x"]},
                    {"id": 1, "op": "atom", "predicate": "p1", "terms": ["x"]},
                    {"id": 2, "op": "implies", "children": [0, 1]},
                    {"id": 3, "op": "forall", "vars": ["x"], "children": [2]},
                ], 3)},
                {"source_index": 1, "formula": formula([
                    {"id": 0, "op": "atom", "predicate": "p1", "terms": ["x"]},
                    {"id": 1, "op": "atom", "predicate": "p2", "terms": ["x"]},
                    {"id": 2, "op": "implies", "children": [0, 1]},
                    {"id": 3, "op": "forall", "vars": ["x"], "children": [2]},
                ], 3)},
                {"source_index": 2, "formula": formula([
                    {"id": 0, "op": "atom", "predicate": "p2", "terms": ["e0"]},
                    {"id": 1, "op": "not", "children": [0]},
                ], 1)},
            ],
            "target": formula([
                {"id": 0, "op": "atom", "predicate": "p0", "terms": ["e0"]}
            ], 0),
            "options": [],
        }}
        compiled = compile_flat_ast_translation(
            raw, premise_count=3, query_type="yes_no_uncertain", option_count=0
        )
        result = run_symbolic_solver(
            {"query_type": "yes_no_uncertain", "intent": "verify_true"},
            {
                "predicates": list(compiled.predicates),
                "functions": [],
                "premises_fol": list(compiled.premises_fol),
                "target_fol": compiled.target_fol,
                "options_fol": [],
            },
        )
        self.assertEqual(result["verdict"], "False")
        self.assertNotIn("$defs", logic_flat_ast_json_schema())

    def test_complex_fallback_uses_one_bounded_semantic_parse_then_z3(self):
        class FakeSemanticParser:
            enabled = True

            def __init__(self):
                self.calls = []

            def chat_json(self, **kwargs):
                self.calls.append(kwargs)
                return {"translation": {
                    "entities": [{"id": "e0", "mentions": ["Vehicle Delta"]}],
                    "predicates": [
                        {"id": "p0", "mentions": ["passes safety test"], "arity": 1},
                        {"id": "p1", "mentions": ["deployment permit"], "arity": 1},
                        {"id": "p2", "mentions": ["certified LiDAR"], "arity": 1},
                    ],
                    "premises": [
                        {"source_index": 0, "ast": {"op": "forall", "vars": ["x"], "body": {
                            "op": "implies", "left": atom("p0", "x"), "right": atom("p1", "x")
                        }}},
                        {"source_index": 1, "ast": {"op": "forall", "vars": ["x"], "body": {
                            "op": "implies", "left": atom("p1", "x"), "right": atom("p2", "x")
                        }}},
                        {"source_index": 2, "ast": {"op": "not", "args": [atom("p2", "e0")]}},
                    ],
                    "target": atom("p0", "e0"),
                    "options": [],
                }}

        fake = FakeSemanticParser()
        pipeline = object.__new__(LogicPipeline)
        pipeline.llm = fake
        pipeline.max_retries = 2
        pipeline._symbolic_ast_cache = OrderedDict()
        pipeline._symbolic_ast_cache_lock = threading.RLock()
        pipeline._symbolic_ast_cache_capacity = 8
        request = dict(
            question="Does Vehicle Delta pass the safety test?",
            premises_nl=[
                "Every vehicle passing the safety test gets a deployment permit.",
                "No vehicle gets a deployment permit unless it has certified LiDAR.",
                "Vehicle Delta does not have certified LiDAR.",
            ],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
            deadline=time.monotonic() + 55.0,
        )
        result = pipeline._answer_with_symbolic_ast(**request)
        self.assertEqual(result.answer, "No")
        self.assertEqual(result.source, "symbolic-ast-z3")
        self.assertIn("P1:", result.explanation)
        self.assertEqual(result.reasoning["type"], "symbolic_entailment")
        self.assertEqual(result.reasoning["method"], "query_unsat")
        self.assertEqual(len(fake.calls), 1)
        self.assertLessEqual(fake.calls[0]["request_timeout_s"], 28.0)
        self.assertEqual(fake.calls[0]["json_schema"], logic_prefix_ir_json_schema())
        cached_result = pipeline._answer_with_symbolic_ast(**request)
        self.assertEqual(cached_result.answer, "No")
        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(cached_result.metadata["symbolic_cache_hit"])

    def test_guided_timeout_retries_once_with_same_validated_prefix_contract(self):
        class TimeoutThenPrefix:
            enabled = True

            def __init__(self):
                self.calls = []

            def chat_json(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    raise LLMError("LLM request failed: timed out")
                return {"translation": {
                    "entities": [{"id": "e0", "mentions": ["Alpha"]}],
                    "predicates": [{"id": "p0", "mentions": ["ready"], "arity": 1}],
                    "premises": [{"source_index": 0, "prefix": ["atom", "p0", "e0"]}],
                    "target_prefix": ["atom", "p0", "e0"],
                    "options_prefix": [],
                }}

        fake = TimeoutThenPrefix()
        pipeline = object.__new__(LogicPipeline)
        pipeline.llm = fake
        pipeline.max_retries = 2
        pipeline._symbolic_ast_cache = OrderedDict()
        pipeline._symbolic_ast_cache_lock = threading.RLock()
        pipeline._symbolic_ast_cache_capacity = 8
        result = pipeline._answer_with_symbolic_ast(
            question="Is Alpha ready?",
            premises_nl=["Alpha is ready."],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
            deadline=time.monotonic() + 55,
        )
        self.assertEqual(result.answer, "Yes")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["json_schema"], logic_prefix_ir_json_schema())
        self.assertIsNone(fake.calls[1]["json_schema"])
        self.assertLessEqual(fake.calls[1]["request_timeout_s"], 20.0)

    def test_controlled_disjunction_bypasses_llm_orchestration_and_retrieval(self):
        class FailIndex:
            def search(self, *args, **kwargs):
                raise AssertionError("complex symbolic route must not run retrieval")

        class FakeSemanticParser:
            enabled = True

            def __init__(self):
                self.calls = 0

            def chat_json(self, **kwargs):
                self.calls += 1
                return {"translation": {
                    "entities": [
                        {"id": "e0", "mentions": ["Alpha"]},
                        {"id": "e1", "mentions": ["Beta"]},
                    ],
                    "predicates": [
                        {"id": "p0", "mentions": ["Alpha is ready"], "arity": 1},
                        {"id": "p1", "mentions": ["Beta is ready"], "arity": 1},
                    ],
                    "premises": [
                        {"source_index": 0, "ast": {"op": "or", "args": [
                            atom("p0", "e0"), atom("p1", "e1")
                        ]}},
                        {"source_index": 1, "ast": {"op": "not", "args": [atom("p0", "e0")]}},
                    ],
                    "target": {"op": "none"},
                    "options": [atom("p0", "e0"), atom("p1", "e1")],
                }}

        fake = FakeSemanticParser()
        pipeline = object.__new__(LogicPipeline)
        pipeline.llm = fake
        pipeline.max_retries = 2
        pipeline.predicate_schema = SimpleNamespace(aliases={})
        pipeline.exact_by_id = {}
        pipeline.exact_by_full_key = {}
        pipeline.exact_by_question = {}
        pipeline.index = FailIndex()
        pipeline.reranker = None
        pipeline.low_match_threshold = 0.5
        pipeline._symbolic_ast_cache = OrderedDict()
        pipeline._symbolic_ast_cache_lock = threading.RLock()
        pipeline._symbolic_ast_cache_capacity = 8
        pipeline._orchestrate_query = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complex symbolic route must not call orchestration")
        )

        result = pipeline.answer({
            "premises-NL": [
                "Either Alpha is ready or Beta is ready.",
                "Alpha is not ready.",
            ],
            "question": "Which statement follows?",
            "options": ["Alpha is ready", "Beta is ready"],
        })
        self.assertEqual(result.answer, "B")
        self.assertEqual(result.source, "deterministic-horn")
        self.assertEqual(fake.calls, 0)

    def test_disjunction_and_explicit_negation_prove_correct_branch(self):
        for entity in ("server_omega", "node_kappa"):
            raw = {"translation": {
                "premises": [
                    {"source_index": 0, "ast": {
                        "op": "forall", "vars": ["x"], "body": {
                            "op": "or", "args": [
                                atom("energy_efficient", "x"),
                                atom("requires_cooling", "x"),
                            ]
                        }
                    }},
                    {"source_index": 1, "ast": {
                        "op": "forall", "vars": ["x"], "body": {
                            "op": "implies",
                            "left": atom("energy_efficient", "x"),
                            "right": atom("uses_arm", "x"),
                        }
                    }},
                    {"source_index": 2, "ast": {
                        "op": "forall", "vars": ["x"], "body": {
                            "op": "implies",
                            "left": atom("requires_cooling", "x"),
                            "right": atom("uses_pumps", "x"),
                        }
                    }},
                    {"source_index": 3, "ast": {
                        "op": "not", "args": [atom("uses_arm", entity)]
                    }},
                ],
                "target": {"op": "none"},
                "options": [
                    {"op": "not", "args": [atom("requires_cooling", entity)]},
                    atom("uses_pumps", entity),
                    atom("energy_efficient", entity),
                ],
            }}
            result = solve(raw, 4, "multiple_choice", 3)
            self.assertEqual(result["best_option"], "B")

    def test_nested_existential_binding_is_preserved(self):
        raw = {"translation": {
            "premises": [
                {"source_index": 0, "ast": {
                    "op": "forall", "vars": ["x"], "body": {
                        "op": "implies",
                        "left": atom("researcher", "x"),
                        "right": {
                            "op": "exists", "vars": ["y"], "body": {
                                "op": "and", "args": [
                                    atom("lab_access", "x", "y"),
                                    atom("secure_facility", "y"),
                                ]
                            }
                        },
                    }
                }},
                {"source_index": 1, "ast": atom("researcher", "asha")},
            ],
            "target": {
                "op": "exists", "vars": ["f"], "body": {
                    "op": "and", "args": [
                        atom("lab_access", "asha", "f"),
                        atom("secure_facility", "f"),
                    ]
                }
            },
            "options": [],
        }}
        result = solve(raw, 2, "yes_no_uncertain", 0, intent="verify_true")
        self.assertEqual(result["verdict"], "True")

    def test_biconditional_chain_proves_target(self):
        raw = {"translation": {
            "premises": [
                {"source_index": 0, "ast": {
                    "op": "forall", "vars": ["x"], "body": {
                        "op": "iff",
                        "left": atom("autonomous_watering", "x"),
                        "right": atom("soil_dry", "x"),
                    }
                }},
                {"source_index": 1, "ast": {
                    "op": "forall", "vars": ["x"], "body": {
                        "op": "implies",
                        "left": atom("heatwave_sensor_active", "x"),
                        "right": atom("soil_dry", "x"),
                    }
                }},
                {"source_index": 2, "ast": atom("heatwave_sensor_active", "greenhouse_basil")},
            ],
            "target": atom("autonomous_watering", "greenhouse_basil"),
            "options": [],
        }}
        result = solve(raw, 3, "yes_no_uncertain", 0, intent="verify_true")
        self.assertEqual(result["verdict"], "True")

    def test_unsat_theory_selects_meta_inconsistency_option(self):
        raw = {"translation": {
            "premises": [
                {"source_index": 0, "ast": {
                    "op": "forall", "vars": ["x"], "body": {
                        "op": "implies",
                        "left": atom("secure_site", "x"),
                        "right": {
                            "op": "exists", "vars": ["g"], "body": {
                                "op": "and", "args": [
                                    atom("generator", "g"),
                                    atom("belongs_to", "g", "x"),
                                ]
                            }
                        },
                    }
                }},
                {"source_index": 1, "ast": atom("secure_site", "harbor")},
                {"source_index": 2, "ast": {
                    "op": "not", "args": [{
                        "op": "exists", "vars": ["g"], "body": {
                            "op": "and", "args": [
                                atom("generator", "g"),
                                atom("belongs_to", "g", "harbor"),
                            ]
                        }
                    }]
                }},
            ],
            "target": {"op": "none"},
            "options": [
                atom("all_generators_passed", "harbor"),
                {"op": "theory_inconsistent"},
                atom("needs_generator", "harbor"),
            ],
        }}
        result = solve(raw, 3, "multiple_choice", 3)
        self.assertEqual(result["best_option"], "B")
        self.assertIn("UNSAT", result["explanation"])

    def test_strongest_conclusion_uses_formula_implication_not_unsat_core_size(self):
        raw = {"translation": {
            "premises": [
                {"source_index": 0, "ast": atom("p", "a")},
                {"source_index": 1, "ast": atom("q", "a")},
            ],
            "target": {"op": "none"},
            "options": [
                atom("p", "a"),
                {"op": "and", "args": [atom("p", "a"), atom("q", "a")]},
            ],
        }}
        result = solve(
            raw, 2, "multiple_choice", 2,
            intent="choose_strongest_conclusion",
        )
        self.assertEqual(result["best_option"], "B")

    def test_multiple_supported_choose_true_options_are_reported_ambiguous(self):
        raw = {"translation": {
            "premises": [
                {"source_index": 0, "ast": atom("p", "a")},
                {"source_index": 1, "ast": atom("q", "a")},
            ],
            "target": {"op": "none"},
            "options": [atom("p", "a"), atom("q", "a")],
        }}
        result = solve(raw, 2, "multiple_choice", 2, intent="choose_true")
        self.assertEqual(result["verdict"], "Uncertain")
        self.assertNotIn("best_option", result)

    def test_lexical_negative_predicate_is_canonicalized(self):
        raw = {"translation": {
            "premises": [
                {"source_index": 0, "ast": atom("does_not_have_lidar", "delta")},
                {"source_index": 1, "ast": {
                    "op": "forall", "vars": ["x"], "body": {
                        "op": "implies",
                        "left": atom("permit", "x"),
                        "right": atom("has_lidar", "x"),
                    }
                }},
            ],
            "target": {"op": "not", "args": [atom("permit", "delta")]},
            "options": [],
        }}
        result = solve(raw, 2, "yes_no_uncertain", 0, intent="verify_true")
        self.assertEqual(result["verdict"], "True")

    def test_missing_target_and_wrong_option_count_are_rejected(self):
        raw = {"translation": {
            "premises": [{"source_index": 0, "ast": atom("p", "a")}],
            "target": {"op": "none"},
            "options": [],
        }}
        with self.assertRaises(AstValidationError):
            compile_ast_translation(
                raw, premise_count=1, query_type="yes_no_uncertain", option_count=0
            )

    def test_inconsistency_meta_operator_is_rejected_inside_a_premise(self):
        raw = {"translation": {
            "premises": [
                {"source_index": 0, "ast": {"op": "theory_inconsistent"}}
            ],
            "target": atom("ready", "alpha"),
            "options": [],
        }}
        with self.assertRaisesRegex(AstValidationError, "premise 0"):
            compile_ast_translation(
                raw, premise_count=1, query_type="yes_no_uncertain", option_count=0
            )

    def test_schema_and_validator_bound_quantifier_variable_arrays(self):
        schema = logic_ast_json_schema()
        self.assertEqual(schema["$defs"]["expression"]["properties"]["vars"]["maxItems"], 16)
        raw = {"translation": {
            "premises": [{
                "source_index": 0,
                "ast": {
                    "op": "forall",
                    "vars": [f"entity_id_{index}" for index in range(400)],
                    "body": atom("ready", "x"),
                },
            }],
            "target": atom("ready", "alpha"),
            "options": [],
        }}
        with self.assertRaisesRegex(AstValidationError, "maximum is 16"):
            compile_ast_translation(
                raw, premise_count=1, query_type="yes_no_uncertain", option_count=0
            )

    def test_validator_rejects_invented_unused_quantifier_variables(self):
        raw = {"translation": {
            "premises": [{
                "source_index": 0,
                "ast": {
                    "op": "forall",
                    "vars": ["x", "entity_id_0"],
                    "body": atom("ready", "x"),
                },
            }],
            "target": atom("ready", "alpha"),
            "options": [],
        }}
        with self.assertRaisesRegex(AstValidationError, "unused in its body"):
            compile_ast_translation(
                raw, premise_count=1, query_type="yes_no_uncertain", option_count=0
            )

    def test_symbol_tables_prevent_entity_and_predicate_drift(self):
        raw = {"translation": {
            "entities": [{"id": "e0", "mentions": ["Model Med-V4", "Med-V4"]}],
            "predicates": [{"id": "p0", "mentions": ["trained on medical data"], "arity": 1}],
            "premises": [{"source_index": 0, "ast": atom("p0", "e0")}],
            "target": atom("p0", "e0"),
            "options": [],
        }}
        compiled = compile_ast_translation(
            raw, premise_count=1, query_type="yes_no_uncertain", option_count=0
        )
        self.assertEqual(compiled.premises_fol, ("p0(e0)",))

        raw["translation"]["target"] = atom("trained_on_medical_data", "med_v4")
        with self.assertRaisesRegex(AstValidationError, "undeclared"):
            compile_ast_translation(
                raw, premise_count=1, query_type="yes_no_uncertain", option_count=0
            )


if __name__ == "__main__":
    unittest.main()
