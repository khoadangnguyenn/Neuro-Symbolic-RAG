import unittest
from unittest.mock import patch
import json
import socket

from exact_pipeline.engines.symbolic_solver import run_symbolic_solver
from exact_pipeline.engines.logic import (
    _extract_symbolic_options,
    _validate_logic_answer_shape,
    _validate_symbolic_translation_contract,
)
from exact_pipeline.llm.llm import (
    _apply_qwen_thinking_mode,
    _qwen_hard_thinking_payload,
    _structured_output_payload,
    OpenAICompatibleLLM,
    LLMError,
    parse_llm_response,
)


class QwenAndSymbolicGuardsTest(unittest.TestCase):
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '{"answer":"A"}'}}]}
            ).encode()

    def test_parse_json_inside_think_when_content_would_be_empty(self):
        parsed = parse_llm_response('<think>{"glossary":{"cat(x)":"x is a cat"}}</think>')
        self.assertEqual(parsed["glossary"]["cat(x)"], "x is a cat")

    def test_qwen3_payload_uses_hard_no_think_switch(self):
        payload = _qwen_hard_thinking_payload("Qwen/Qwen3-8B", thinking=False)
        self.assertEqual(payload["chat_template_kwargs"]["enable_thinking"], False)

    def test_qwen3_cannot_be_reenabled_by_a_fallback_caller(self):
        payload = _qwen_hard_thinking_payload("Qwen/Qwen3-8B", thinking=True)
        system, user = _apply_qwen_thinking_mode("system /think", "user", thinking=True)
        self.assertEqual(payload["chat_template_kwargs"]["enable_thinking"], False)
        self.assertNotIn("/think", system.replace("/no_think", ""))
        self.assertTrue(system.endswith("/no_think"))
        self.assertTrue(user.endswith("/no_think"))

    def test_json_schema_is_forwarded_as_structured_output_contract(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        payload = _structured_output_payload(schema)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(
            payload["response_format"]["json_schema"]["schema"], schema
        )

    def test_chat_json_sends_schema_while_plain_chat_remains_valid(self):
        requests = []
        timeouts = []

        def fake_urlopen(request, timeout):
            requests.append(json.loads(request.data.decode()))
            timeouts.append(timeout)
            return self._FakeResponse()

        client = OpenAICompatibleLLM("http://localhost:8001", "Qwen3-8B")
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertEqual(
                client.chat(system_prompt="system", user_prompt="user")["answer"], "A"
            )
            self.assertEqual(
                client.chat_json(
                    system_prompt="system", user_prompt="user", json_schema=schema,
                    request_timeout_s=7.0,
                )["answer"],
                "A",
            )

        self.assertNotIn("response_format", requests[0])
        self.assertEqual(
            requests[1]["response_format"]["json_schema"]["schema"], schema
        )
        self.assertFalse(
            requests[1]["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertEqual(timeouts[1], 7.0)

    def test_chat_json_converts_socket_timeout_to_llm_error(self):
        client = OpenAICompatibleLLM("http://localhost:8001", "Qwen3-8B")
        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            with self.assertRaises(LLMError):
                client.chat_json(
                    system_prompt="system",
                    user_prompt="user",
                    request_timeout_s=0.1,
                )

    def test_time_literal_does_not_break_fol_parser(self):
        result = run_symbolic_solver(
            {"query_type": "multiple_choice", "intent": "choose_true"},
            {
                "predicates": ["departs_at_time(x)"],
                "functions": [],
                "premises_fol": ["departs_at_time(flight_z7, 19:00)"],
                "options_fol": ["departs_at_time(flight_z7, 21:00)"],
            },
        )
        self.assertNotEqual(result["verdict"], "Error")

    def test_bare_numeric_constant_is_typed_as_number(self):
        result = run_symbolic_solver(
            {"query_type": "yes_no_uncertain", "intent": "verify_true"},
            {
                "predicates": ["meeting_room(x)", "owned_by_club(x)"],
                "functions": [],
                "premises_fol": [
                    "∀x((meeting_room(x) ∧ owned_by_club(x)) -> (count_of_meeting_rooms = 3))",
                    "meeting_room(room_a)",
                    "owned_by_club(room_a)",
                ],
                "target_fol": "count_of_meeting_rooms = 3",
                "options_fol": [],
            },
        )
        self.assertEqual(result["verdict"], "True")

    def test_hyphenated_entity_is_normalized_before_parsing(self):
        result = run_symbolic_solver(
            {"query_type": "yes_no_uncertain", "intent": "verify_true"},
            {
                "predicates": ["ready(x)"],
                "functions": [],
                "premises_fol": ["ready(MedKit-7)"],
                "target_fol": "ready(MedKit-7)",
                "options_fol": [],
            },
        )
        self.assertEqual(result["verdict"], "True")

    def test_inconsistent_predicate_signature_is_rejected(self):
        result = run_symbolic_solver(
            {"query_type": "yes_no_uncertain", "intent": "verify_true"},
            {
                "predicates": ["linked(x)"],
                "functions": [],
                "premises_fol": ["linked(node_a)", "linked(node_a, node_b)"],
                "target_fol": "linked(node_a)",
                "options_fol": [],
            },
        )
        self.assertEqual(result["verdict"], "Error")
        self.assertIn("inconsistent signature", result["explanation"])

    def test_predicate_name_does_not_retype_same_spelling_as_constant(self):
        result = run_symbolic_solver(
            {"query_type": "multiple_choice", "intent": "verify_true"},
            {
                # A noisy glossary may hallucinate today(x), while the formulas
                # correctly use today as an object constant.
                "predicates": ["rain(x)", "wet(x)", "today(x)", "snowing(x)"],
                "functions": [],
                "premises_fol": ["∀x(rain(x) → wet(x))", "rain(today)"],
                "target_fol": "",
                "options_fol": ["¬wet(today)", "wet(today)", "snowing(today)"],
            },
        )
        self.assertEqual(result["verdict"], "True")
        self.assertEqual(result["best_option"], "B")

    def test_logic_answer_contract_rejects_malformed_numeric_output(self):
        question = "How many entrances does the venue have?"
        self.assertIsNone(_validate_logic_answer_shape("3, x", question))
        self.assertEqual(_validate_logic_answer_shape("4", question), "4")

    def test_symbolic_options_are_read_from_payload_when_not_inline(self):
        self.assertEqual(
            _extract_symbolic_options(
                "Which conclusion follows?",
                ["A. Alpha is active.", "B. Alpha is inactive."],
            ),
            ["Alpha is active.", "Alpha is inactive."],
        )

    def test_logic_answer_contract_rejects_unverified_explanation_as_value(self):
        question = "At what time does Flight Q8 depart?"
        self.assertIsNone(
            _validate_logic_answer_shape(
                "directly given in the premises: departs_at(q8, 18_40)", question
            )
        )
        self.assertEqual(_validate_logic_answer_shape("18:40", question), "18:40")

    def test_symbolic_contract_rejects_dropped_premise_and_empty_target(self):
        errors = _validate_symbolic_translation_contract(
            {
                "premises_fol": ["ready(alpha)"],
                "target_fol": "",
                "options_fol": [],
            },
            premise_count=2,
            query_type="yes_no_uncertain",
            option_count=0,
        )
        self.assertTrue(any("exactly 2" in error for error in errors))
        self.assertTrue(any("non-empty target_fol" in error for error in errors))

    def test_symbolic_contract_rejects_wrong_multiple_choice_arity(self):
        errors = _validate_symbolic_translation_contract(
            {
                "premises_fol": ["ready(alpha)"],
                "target_fol": "",
                "options_fol": ["ready(alpha)"],
            },
            premise_count=1,
            query_type="multiple_choice",
            option_count=3,
        )
        self.assertTrue(any("exactly 3 options_fol" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
