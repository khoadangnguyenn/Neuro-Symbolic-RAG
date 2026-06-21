import tempfile
import unittest
from pathlib import Path

from exact_pipeline.core.models import LogicExample
from exact_pipeline.engines.horn_reasoner import try_deterministic_logic
from exact_pipeline.engines.schema_learner import PredicateSchemaLearner


class PredicateSchemaLearnerTest(unittest.TestCase):
    def _training_example(self):
        return LogicExample(
            record_id="schema-1",
            question_id="schema-1-q1",
            premises_nl=["Mira bought a fare pass."],
            premises_fol=["has_transit_ticket(Mira)"],
            question="An unrelated training question",
            answer="a label that must never be used",
            explanation="also ignored",
            premise_indices=[],
        )

    def test_aligned_data_learns_schema_without_answer_labels(self):
        learner = PredicateSchemaLearner.from_examples([self._training_example()])
        self.assertEqual(learner.aliases["bought_fare_pass"], "has_transit_ticket")
        self.assertNotIn("a label", " ".join(learner.aliases.values()))

        result = try_deterministic_logic(
            question="May Mira enter the station?",
            premises_nl=[
                "If a commuter has a transit ticket, then that commuter may enter the station.",
                "Mira bought a fare pass.",
            ],
            premises_fol=[],
            options=["Yes", "No", "Uncertain"],
            query_type="yes_no_uncertain",
            intent="verify_true",
            learned_aliases=learner.aliases,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "Yes")

    def test_schema_cache_round_trip_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "predicate_schema.json"
            first = PredicateSchemaLearner.from_examples(
                [self._training_example()], cache_path=cache
            )
            second = PredicateSchemaLearner.from_examples(
                [self._training_example()], cache_path=cache
            )
            self.assertTrue(cache.exists())
            self.assertEqual(first.aliases, second.aliases)
            self.assertEqual(first.evidence, second.evidence)


if __name__ == "__main__":
    unittest.main()
