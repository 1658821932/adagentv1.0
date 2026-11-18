"""The mock backend must satisfy the same output contract as the real analysis."""
import subprocess
import sys
import unittest
from importlib import import_module
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.registry import call_tool  # noqa: E402

OUTPUT = import_module("tools.tool01_failure_analyzer.schemas").Output
FIELDS = set(OUTPUT.model_fields)


class MockBackendTests(unittest.TestCase):
    def call(self, **arguments):
        result = call_tool("failure_analyzer", arguments, mode="mock")
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["tool"], "failure_analyzer")
        OUTPUT.model_validate(result["result"])
        return result["result"]

    def test_minimal_token_input(self):
        result = self.call(token="scene-token")
        self.assertEqual(set(result), FIELDS)
        self.assertEqual(result["token"], "scene-token")
        self.assertIsNone(result["simulation"])
        self.assertNotIn("failure_analysis", result)

    def test_input_with_simulation(self):
        result = self.call(token="scene-token", simulation="experiment/batch")
        self.assertEqual(result["simulation"], "experiment/batch")
        self.assertEqual(result["token"], "scene-token")
        OUTPUT.model_validate(result)

    def test_values_are_deterministic(self):
        self.assertEqual(self.call(token="scene-token"), self.call(token="scene-token"))

    def test_nested_fields_use_the_current_structure(self):
        result = self.call(token="scene-token")
        self.assertEqual(set(result["failure"]), {"outcome", "interaction", "ego_cause"})
        self.assertEqual(set(result["attack"]),
                         {"map_context", "ego_action", "ego_lane_keeping",
                          "other_origin", "other_action", "other_lane_keeping"})
        self.assertEqual(set(result["environment"]), {"time", "weather", "extra"})
        self.assertEqual(set(result["verification"]), {"sample_frames", "corrections", "notes"})
        self.assertEqual(result["violation_step"], 7)

    def test_mock_reads_no_file_and_calls_no_backend(self):
        with mock.patch("builtins.open", side_effect=AssertionError("mock opened a file")), \
             mock.patch.object(subprocess, "run", side_effect=AssertionError("mock ran a process")), \
             mock.patch("agent.llm_client.LLMClient", side_effect=AssertionError("mock called a model")):
            result = self.call(token="scene-token")
        self.assertEqual(result["token"], "scene-token")


if __name__ == "__main__":
    unittest.main()
