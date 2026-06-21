import unittest
from types import SimpleNamespace

from exact_pipeline.engines.horn_reasoner import assess_horn_capability, try_deterministic_logic
from exact_pipeline.engines.logic import LogicPipeline
from tests.run_diverse_testcases import TESTCASES


class HornReasonerTest(unittest.TestCase):
    def setUp(self):
        self.study_premises = [
            "If a researcher completed ethics training and has lab access, then that researcher can handle participant data.",
            "If a researcher can handle participant data and has supervisor approval, then that researcher may join Study Alpha.",
            "Every researcher who may join Study Alpha is listed as an active contributor.",
            "Asha completed ethics training.",
            "Asha has lab access.",
            "Asha has supervisor approval.",
            "Study Alpha has 12 enrolled participants.",
            "No premise states whether Asha has budget approval.",
        ]

    def solve(self, question, query_type, options=(), intent="open_analysis"):
        result = try_deterministic_logic(
            question=question,
            premises_nl=self.study_premises,
            premises_fol=[],
            options=list(options),
            query_type=query_type,
            intent=intent,
        )
        self.assertIsNotNone(result)
        return result

    def test_multiple_choice_uses_forward_chain(self):
        result = self.solve(
            "Based on the premises, which option is logically supported?\n"
            "A. Asha may join Study Alpha\n"
            "B. Asha cannot handle participant data\n"
            "C. Asha has budget approval\n"
            "D. Study Alpha has 20 enrolled participants",
            "multiple_choice",
            ["A", "B", "C", "D"],
            "choose_true",
        )
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.premises_used, [0, 1, 3, 4, 5])

    def test_yes_no_uncertain_is_final_when_not_entailed(self):
        result = self.solve(
            "Does Asha have budget approval?",
            "yes_no_uncertain",
            ["Yes", "No", "Uncertain"],
        )
        self.assertEqual(result.answer, "Uncertain")
        self.assertEqual(result.premises_used, [7])

    def test_numeric_open_ended_extracts_value_and_premise(self):
        result = self.solve(
            "How many enrolled participants does Study Alpha have?",
            "open_ended",
        )
        self.assertEqual(result.answer, "12")
        self.assertEqual(result.premises_used, [6])

    def test_text_open_ended_returns_derived_entity(self):
        result = self.solve(
            "Which researcher may join Study Alpha?",
            "open_ended",
        )
        self.assertEqual(result.answer, "Asha")
        self.assertEqual(result.premises_used, [0, 1, 3, 4, 5])

    def test_public_pipeline_routes_all_five_answer_shapes_without_llm(self):
        """The API entry point must keep the supported fragment off slow fallbacks."""

        pipeline = object.__new__(LogicPipeline)
        pipeline.predicate_schema = SimpleNamespace(aliases={})
        cases = [
            (
                {
                    "premises": self.study_premises,
                    "query": (
                        "Based on the premises, which option is logically supported?\n"
                        "A. Asha may join Study Alpha\n"
                        "B. Asha cannot handle participant data\n"
                        "C. Asha has budget approval\n"
                        "D. Study Alpha has 20 enrolled participants"
                    ),
                    "options": ["A", "B", "C", "D"],
                },
                "A",
                [0, 1, 3, 4, 5],
            ),
            (
                {
                    "premises": self.study_premises,
                    "query": "Is Asha listed as an active contributor?",
                    "options": ["Yes", "No", "Uncertain"],
                },
                "Yes",
                [0, 1, 2, 3, 4, 5],
            ),
            (
                {
                    "premises": self.study_premises,
                    "query": "Does Asha have budget approval?",
                    "options": ["Yes", "No", "Uncertain"],
                },
                "Uncertain",
                [7],
            ),
            (
                {
                    "premises": self.study_premises,
                    "query": "How many enrolled participants does Study Alpha have?",
                    "options": [],
                },
                "12",
                [6],
            ),
            (
                {
                    "premises": self.study_premises,
                    "query": "Which researcher may join Study Alpha?",
                    "options": [],
                },
                "Asha",
                [0, 1, 3, 4, 5],
            ),
        ]

        for payload, expected_answer, expected_support in cases:
            with self.subTest(query=payload["query"]):
                result = pipeline.answer(payload)
                self.assertEqual(result.answer, expected_answer)
                self.assertEqual(result.premises_used, expected_support)
                self.assertIn(
                    result.source,
                    {"deterministic-horn", "deterministic-grounded-value"},
                )
                self.assertEqual(result.metadata["symbolic_stage"], "pre_routing")

    def test_reported_advanced_cases_use_grammar_and_grounded_explanations(self):
        expected = {
            "Type 1 - Advanced Logic (Transitive Disjunction and Excluded Middle)": ("B", [0, 1, 2, 3]),
            "Type 1 - Advanced Logic (Complex Chain with Biconditional Equivalence)": ("Yes", [0, 1, 2]),
            "Type 1 - Advanced Logic (Nested Quantifiers and Vacuous Truth Trap)": ("B", [0, 2, 3]),
            "hard_type1_modus_tollens": ("No", [0, 1, 2]),
            "hard_type1_disjunctive_syllogism": ("Yes", [0, 1, 2, 3]),
            "hard_type1_nested_numeric": ("45", [0, 1, 3]),
            "hard_type1_text_extraction": ("Bob", [0, 1, 4]),
        }
        pipeline = object.__new__(LogicPipeline)
        pipeline.predicate_schema = SimpleNamespace(aliases={})
        seen = set()

        for testcase in TESTCASES:
            raw = testcase.get("request_payload")
            name = raw.get("query_id") if raw else testcase.get("name")
            if name not in expected:
                continue
            seen.add(name)
            payload = (
                {
                    "premises": raw.get("premises", []),
                    "query": raw.get("query", ""),
                    "options": raw.get("options", []),
                }
                if raw else testcase["payload"]
            )
            with self.subTest(name=name):
                result = pipeline.answer(payload)
                answer, support = expected[name]
                self.assertEqual(result.answer, answer)
                self.assertEqual(result.premises_used, support)
                self.assertTrue(result.explanation)
                for index in support:
                    self.assertIn(f"P{index + 1}:", result.explanation)
                self.assertNotIn("LLM", result.source)

        self.assertEqual(seen, set(expected))

    def test_single_antecedent_contraposition(self):
        result = try_deterministic_logic(
            question=(
                "Based on the above premises, which of the following is true?\n"
                "A. Drone X has a high-quality camera.\n"
                "B. Drone X does not have a high-quality camera."
            ),
            premises_nl=[
                "If a drone has a high-quality camera, it has long battery life.",
                "Drone X does not have long battery life.",
            ],
            premises_fol=[],
            options=[],
            query_type="multiple_choice",
            intent="choose_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "B")
        self.assertEqual(result.premises_used, [0, 1])

    def test_universal_class_entailment(self):
        result = try_deterministic_logic(
            question="Is Fluffy a mammal?",
            premises_nl=[
                "All cats are mammals.",
                "Fluffy is a cat.",
            ],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="open_analysis",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Yes")
        self.assertEqual(result.premises_used, [0, 1])

    def test_fewest_premises_uses_minimal_proof_support(self):
        result = try_deterministic_logic(
            question=(
                "Based on the above premises, which is the fewest in the following premises?\n"
                "A. B is true\n"
                "B. C is true\n"
                "C. D is true"
            ),
            premises_nl=[
                "If A then B.",
                "If B then C.",
                "If A then C.",
                "A is true.",
            ],
            premises_fol=[],
            options=[],
            query_type="multiple_choice",
            intent="choose_fewest_premises",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.premises_used, [0, 3])

    def test_opaque_fol_yes_no_options_use_symbolic_closure(self):
        result = try_deterministic_logic(
            question="Is Fluffy a mammal?\nA. Yes\nB. No\nC. Uncertain",
            premises_nl=[],
            premises_fol=[
                "∀x (C(x) → M(x))",
                "C(Fluffy)",
            ],
            options=[],
            query_type="multiple_choice",
            intent="open_analysis",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.premises_used, [0, 1])

    def test_domain_role_prefixes_do_not_block_forward_chain(self):
        result = try_deterministic_logic(
            question=(
                "Based on the premises, which option is logically supported?\n"
                "A. Nova Supplies receives a certification review\n"
                "B. Nova Supplies failed inspection\n"
                "C. Nova Supplies exports products internationally\n"
                "D. The review queue contains 50 vendors"
            ),
            premises_nl=[
                "If a vendor submitted all compliance documents and passed inspection, then that vendor is eligible for certification.",
                "If a vendor is eligible for certification and paid certification fee, then that vendor receives a certification review.",
                "Every vendor receiving a certification review appears in the review queue.",
                "No premise states whether Nova Supplies failed inspection.",
                "Nova Supplies submitted all compliance documents.",
                "Nova Supplies passed inspection.",
                "Nova Supplies paid certification fee.",
                "The review queue currently has 23 vendors.",
            ],
            premises_fol=[],
            options=[],
            query_type="multiple_choice",
            intent="choose_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.premises_used, [0, 1, 4, 5, 6])

    def test_inflected_verbs_match_member_rules(self):
        result = try_deterministic_logic(
            question=(
                "Based on the premises, which option is logically supported?\n"
                "A. Minh may vote in club elections\n"
                "B. Minh is not in good standing\n"
                "C. Minh serves on a committee\n"
                "D. The club owns ten meeting rooms"
            ),
            premises_nl=[
                "If a member paid annual dues and signed conduct agreement, then that member is in good standing.",
                "If a member is in good standing and attended orientation, then that member may vote in club elections.",
                "Every member who may vote in club elections is listed on the voter roll.",
                "Minh paid annual dues.",
                "Minh signed conduct agreement.",
                "Minh attended orientation.",
                "The club owns three meeting rooms.",
            ],
            premises_fol=[],
            options=[],
            query_type="multiple_choice",
            intent="choose_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.premises_used, [0, 1, 3, 4, 5])

    def test_schema_is_induced_for_unseen_paraphrase_and_article_shaped_name(self):
        result = try_deterministic_logic(
            question=(
                "Based on the premises, which option is logically supported?\n"
                "A. An receives priority support\n"
                "B. An cannot submit support tickets\n"
                "C. An purchased consulting services\n"
                "D. The company offers ten subscription plans"
            ),
            premises_nl=[
                "If a customer created an account and verified an email address, then that customer may submit support tickets.",
                "If a customer may submit support tickets and has an active subscription, then that customer receives priority support.",
                "Every customer receiving priority support appears in the priority queue.",
                "An completed account creation.",
                "An verified an email address.",
                "An has an active subscription.",
                "The company offers seven subscription plans.",
                "No premise states whether An purchased consulting services.",
            ],
            premises_fol=[],
            options=[],
            query_type="multiple_choice",
            intent="choose_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "A")
        self.assertEqual(result.premises_used, [0, 1, 3, 4, 5])

    def test_elliptical_conjunction_does_not_merge_passed_with_failed(self):
        result = try_deterministic_logic(
            question=(
                "Based on the premises, which option is logically supported?\n"
                "A. Mai owns a telescope\n"
                "B. The university owns 10 observatories\n"
                "C. Mai failed Physics II\n"
                "D. Mai may access the advanced laboratory"
            ),
            premises_nl=[
                "If a student passed Physics I and Physics II, then that student may enroll in Quantum Mechanics.",
                "If a student may enroll in Quantum Mechanics and paid tuition, then that student may access the advanced laboratory.",
                "Mai passed Physics I.",
                "Mai passed Physics II.",
                "Mai paid tuition.",
                "The university owns 6 observatories.",
                "No premise states whether Mai owns a telescope.",
            ],
            premises_fol=[],
            options=[],
            query_type="multiple_choice",
            intent="choose_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "D")
        self.assertIn(3, result.premises_used)

    def test_roman_numeral_course_versions_are_distinct_predicates(self):
        result = try_deterministic_logic(
            question="May Bao enroll in Differential Equations?",
            premises_nl=[
                "If a student passed Calculus I and passed Calculus II, then that student may enroll in Differential Equations.",
                "Bao passed Calculus I.",
            ],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Uncertain")

    def test_universal_rules_are_induced_from_observed_class_schema(self):
        result = try_deterministic_logic(
            question="Can Khanh access the audit portal?",
            premises_nl=[
                "All certified auditors completed compliance training.",
                "All employees who completed compliance training may access the audit portal.",
                "Khanh is a certified auditor.",
            ],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Yes")

    def test_missing_antecedent_stays_uncertain_under_open_world(self):
        result = try_deterministic_logic(
            question="Is RiverTech listed in the preferred vendor directory?",
            premises_nl=[
                "If a supplier passed inspection and submitted compliance forms, then that supplier is approved.",
                "If a supplier is approved and pays the annual fee, then that supplier is listed in the preferred vendor directory.",
                "RiverTech passed inspection.",
                "RiverTech submitted compliance forms.",
                "No premise states whether RiverTech paid the annual fee.",
            ],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Uncertain")

    def test_contradictory_facts_do_not_pick_an_arbitrary_truth_value(self):
        result = try_deterministic_logic(
            question="Is Nia certified?",
            premises_nl=["Nia is certified.", "Nia is not certified."],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
        )
        self.assertIsNone(result)

    def test_grounded_value_projection_returns_typed_time(self):
        result = try_deterministic_logic(
            question="At what time does Flight Q8 depart?",
            premises_nl=["Flight Q8 departs at 18:40."],
            premises_fol=[],
            options=[],
            query_type="open_ended",
            intent="open_analysis",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "18:40")
        self.assertEqual(result.premises_used, [0])

    def test_relational_object_schema_links_unseen_access_wording(self):
        result = try_deterministic_logic(
            question="Which person may enter the VIP area?",
            premises_nl=[
                "All gold members receive lounge access.",
                "All people with lounge access may enter the VIP area.",
                "Quoc is a gold member.",
            ],
            premises_fol=[],
            options=[],
            query_type="open_ended",
            intent="open_analysis",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Quoc")

    def test_named_entities_with_determiners_hyphens_and_trailing_roles_are_grounded(self):
        result = try_deterministic_logic(
            question="Is AidBox-9 ready to dispatch?",
            premises_nl=[
                "If a package is sealed and its route is clear, then it is ready to dispatch.",
                "If an alternate route is mapped, then the route is clear.",
                "The AidBox-9 package is sealed.",
                "An alternate route is mapped for AidBox-9.",
            ],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Yes")

    def test_lexical_negation_never_aliases_to_positive_property(self):
        result = try_deterministic_logic(
            question="Does Orion lack a safety certificate?",
            premises_nl=["Orion has a safety certificate."],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "No")

    def test_truth_status_and_entailment_status_are_distinct(self):
        premises = [
            "If a device passes inspection and has operator approval, then it may launch.",
            "Unit Sigma passes inspection.",
        ]
        direct = try_deterministic_logic(
            question="May Unit Sigma launch?",
            premises_nl=premises,
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
        )
        proof_status = try_deterministic_logic(
            question="Do the premises establish that Unit Sigma may launch?",
            premises_nl=premises,
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
        )
        self.assertEqual(direct.answer, "Uncertain")
        self.assertEqual(proof_status.answer, "No")

    def test_compound_option_combines_truth_and_non_entailment(self):
        result = try_deterministic_logic(
            question=(
                "Which conclusion follows?\n"
                "A. Relay One is online, but final approval is not established by the premises\n"
                "B. Relay One has final approval"
            ),
            premises_nl=["Relay One is online."],
            premises_fol=[],
            options=[],
            query_type="multiple_choice",
            intent="choose_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "A")

    def test_specific_consequent_projects_only_to_weaker_later_predicate(self):
        result = try_deterministic_logic(
            question="Is Node R resilient?",
            premises_nl=[
                "If a node has backup power, then that node can disconnect from the upstream network.",
                "If a node can disconnect, then that node is resilient.",
                "Node R has backup power.",
            ],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Yes")

    def test_impersonal_weather_context_entails_ground_property(self):
        result = try_deterministic_logic(
            question=(
                "Based on the premises, which statement is true?\n"
                "A. The ground is dry\n"
                "B. The ground is wet\n"
                "C. It is snowing"
            ),
            premises_nl=[
                "If it rains, the ground gets wet.",
                "It is raining today.",
            ],
            premises_fol=[],
            options=["A", "B", "C"],
            query_type="multiple_choice",
            intent="choose_true",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "B")
        self.assertEqual(result.premises_used, [0, 1])
        self.assertEqual(result.metadata["executor"], "deterministic-horn")

    def test_nested_multi_argument_fol_bypasses_horn(self):
        premises_fol = [
            "∀x (Researcher(x) → ∃y (LabAccess(x, y) ∧ SecureFacility(y)))",
            "Researcher(Asha)",
        ]
        capability = assess_horn_capability(
            question="Does Asha have access to any secure facility?",
            premises_nl=[],
            premises_fol=premises_fol,
            options=["Yes", "No", "Uncertain"],
        )
        self.assertFalse(capability.supported)
        self.assertIn("existential_quantifier", capability.reasons)
        self.assertIn("multi_argument_predicate", capability.reasons)
        self.assertIsNone(
            try_deterministic_logic(
                question="Does Asha have access to any secure facility?",
                premises_nl=[],
                premises_fol=premises_fol,
                options=["Yes", "No", "Uncertain"],
                query_type="yes_no_uncertain",
                intent="verify_true",
            )
        )

    def test_controlled_non_horn_surfaces_are_supported_deterministically(self):
        for premises in (
            ["A controller activates if and only if a sensor is active."],
            ["All secure sites have at least one backup generator."],
            ["A node is either active or in maintenance mode."],
        ):
            capability = assess_horn_capability(
                question="What follows?",
                premises_nl=premises,
                premises_fol=[],
                options=[],
            )
            self.assertTrue(capability.supported, premises)

    def test_simple_unary_horn_fragment_remains_on_fast_path(self):
        capability = assess_horn_capability(
            question="Is Fluffy a mammal?",
            premises_nl=["All cats are mammals.", "Fluffy is a cat."],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
        )
        self.assertTrue(capability.supported)


if __name__ == "__main__":
    unittest.main()
