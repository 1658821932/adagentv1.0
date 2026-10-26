import unittest

from agent.schemas import validate_decision


AVAILABLE_TOOLS = {
    "scene_conceiver",
    "video_retriever",
}


class DecisionSchemaTests(unittest.TestCase):
    def test_valid_call_tool(self):
        decision = {
            "action": "call_tool",
            "tool_name": "scene_conceiver",
            "arguments": {"scene_requirement": {}},
            "reason": "Create a high-level scenario plan.",
        }

        valid, errors = validate_decision(decision, AVAILABLE_TOOLS)

        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_valid_finish(self):
        decision = {
            "action": "finish",
            "reason": "The task is complete.",
        }

        valid, errors = validate_decision(decision, AVAILABLE_TOOLS)

        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_missing_tool_name(self):
        decision = {
            "action": "call_tool",
            "arguments": {},
            "reason": "Need another tool.",
        }

        valid, errors = validate_decision(decision, AVAILABLE_TOOLS)

        self.assertFalse(valid)
        self.assertIn("tool_name must be a non-empty string", errors)

    def test_arguments_must_be_object(self):
        decision = {
            "action": "call_tool",
            "tool_name": "scene_conceiver",
            "arguments": "invalid",
            "reason": "Need a scenario plan.",
        }

        valid, errors = validate_decision(decision, AVAILABLE_TOOLS)

        self.assertFalse(valid)
        self.assertIn("arguments must be a JSON object", errors)

    def test_unknown_tool_name(self):
        decision = {
            "action": "call_tool",
            "tool_name": "unknown_tool",
            "arguments": {},
            "reason": "Try another tool.",
        }

        valid, errors = validate_decision(decision, AVAILABLE_TOOLS)

        self.assertFalse(valid)
        self.assertIn("unknown tool_name: unknown_tool", errors)

    def test_invalid_action(self):
        decision = {
            "action": "run_tool",
            "tool_name": "scene_conceiver",
            "arguments": {},
            "reason": "Need a scenario plan.",
        }

        valid, errors = validate_decision(decision, AVAILABLE_TOOLS)

        self.assertFalse(valid)
        self.assertIn("action must be one of: call_tool, finish", errors)

    def test_reason_must_be_non_empty(self):
        decision = {
            "action": "finish",
            "reason": "",
        }

        valid, errors = validate_decision(decision, AVAILABLE_TOOLS)

        self.assertFalse(valid)
        self.assertIn("reason must be a non-empty string", errors)


if __name__ == "__main__":
    unittest.main()
