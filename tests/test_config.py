import json
import tempfile
import unittest
from pathlib import Path

from agent.config import AgentConfig, PROJECT_ROOT
from agent.executor import ToolExecutor


DEFAULT_MODES = {
    "failure_analyzer": "real",
    "scene_requirement_generator": "real",
    "scene_conceiver": "real",
    "video_retriever": "real",
    "instance_discovery": "real",
    "reconstruction": "real",
    "edit_plan_builder": "real",
    "edit_config_generator": "real",
    "scene_editor": "real",
    "scene_validator": "real",
    "policy_runner": "real",
    "failure_case_filter": "real",
    "policy_trainer": "real",
}


class AgentConfigTests(unittest.TestCase):
    def _write_config(self, data):
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "agent.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file)
        return temp_dir, path

    def test_default_config_loads(self):
        config = AgentConfig.load()

        self.assertEqual(config.max_steps, 20)
        self.assertEqual(config.decision_correction_attempts, 1)
        self.assertEqual(config.tool_modes, DEFAULT_MODES)
        self.assertEqual(config.logs_root, PROJECT_ROOT / "logs")

    def test_custom_config_loads(self):
        custom_modes = dict(DEFAULT_MODES)
        custom_modes["video_retriever"] = "mock"
        custom_modes["reconstruction"] = "mock"

        temp_dir, path = self._write_config(
            {
                "agent": {"max_steps": 20},
                "orchestrator": {"decision_correction_attempts": 2},
                "tools": {"modes": custom_modes},
                "logging": {"root_dir": "custom_logs"},
            }
        )
        self.addCleanup(temp_dir.cleanup)

        config = AgentConfig.load(path)

        self.assertEqual(config.max_steps, 20)
        self.assertEqual(config.decision_correction_attempts, 2)
        self.assertEqual(config.tool_modes, custom_modes)
        self.assertEqual(config.logs_root, PROJECT_ROOT / "custom_logs")

    def test_max_steps_must_be_positive_integer(self):
        temp_dir, path = self._write_config(
            {
                "agent": {"max_steps": 0},
                "orchestrator": {"decision_correction_attempts": 1},
                "tools": {"modes": DEFAULT_MODES},
                "logging": {"root_dir": "logs"},
            }
        )
        self.addCleanup(temp_dir.cleanup)

        with self.assertRaisesRegex(
            ValueError,
            "agent.max_steps must be a positive integer",
        ):
            AgentConfig.load(path)

    def test_correction_attempts_must_be_non_negative(self):
        temp_dir, path = self._write_config(
            {
                "agent": {"max_steps": 10},
                "orchestrator": {"decision_correction_attempts": -1},
                "tools": {"modes": DEFAULT_MODES},
                "logging": {"root_dir": "logs"},
            }
        )
        self.addCleanup(temp_dir.cleanup)

        with self.assertRaisesRegex(ValueError, "decision_correction_attempts"):
            AgentConfig.load(path)

    def test_invalid_tool_mode_fails(self):
        modes = dict(DEFAULT_MODES)
        modes["reconstruction"] = "remote"

        temp_dir, path = self._write_config(
            {
                "agent": {"max_steps": 10},
                "orchestrator": {"decision_correction_attempts": 1},
                "tools": {"modes": modes},
                "logging": {"root_dir": "logs"},
            }
        )
        self.addCleanup(temp_dir.cleanup)

        with self.assertRaisesRegex(
            ValueError,
            "tools.modes.reconstruction must be one of: mock, real",
        ):
            AgentConfig.load(path)

    def test_missing_tool_mode_fails(self):
        modes = dict(DEFAULT_MODES)
        modes.pop("reconstruction")

        temp_dir, path = self._write_config(
            {
                "agent": {"max_steps": 10},
                "orchestrator": {"decision_correction_attempts": 1},
                "tools": {"modes": modes},
                "logging": {"root_dir": "logs"},
            }
        )
        self.addCleanup(temp_dir.cleanup)

        with self.assertRaisesRegex(ValueError, "tools.modes is missing tools"):
            AgentConfig.load(path)

    def test_missing_required_fields_fails(self):
        temp_dir, path = self._write_config(
            {
                "agent": {"max_steps": 10},
            }
        )
        self.addCleanup(temp_dir.cleanup)

        with self.assertRaisesRegex(ValueError, "missing required fields"):
            AgentConfig.load(path)

    def test_executor_can_use_custom_modes(self):
        executor = ToolExecutor(tool_modes=DEFAULT_MODES)
        self.assertEqual(executor.tool_modes["video_retriever"], "real")
        self.assertEqual(executor.tool_modes["failure_analyzer"], "real")


if __name__ == "__main__":
    unittest.main()
