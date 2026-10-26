import unittest
from unittest.mock import patch

from agent.executor import ToolExecutor
from agent.schemas import validate_tool_result


class ToolResultSchemaTests(unittest.TestCase):
    def test_valid_success_result(self):
        result = {
            "status": "success",
            "tool": "reconstruction",
            "result": {
                "scene_ply": "outputs/mock/scene.ply",
            },
        }

        valid, errors = validate_tool_result(result, "reconstruction")

        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_valid_error_result(self):
        result = {
            "status": "error",
            "tool": "reconstruction",
            "error": {
                "type": "execution_error",
                "message": "checkpoint not found",
            },
        }

        valid, errors = validate_tool_result(result, "reconstruction")

        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_missing_status(self):
        result = {
            "tool": "reconstruction",
            "result": {},
        }

        valid, errors = validate_tool_result(result, "reconstruction")

        self.assertFalse(valid)
        self.assertIn("status must be one of: success, error", errors)

    def test_tool_name_mismatch(self):
        result = {
            "status": "success",
            "tool": "video_retriever",
            "result": {},
        }

        valid, errors = validate_tool_result(result, "reconstruction")

        self.assertFalse(valid)
        self.assertIn(
            "tool mismatch: expected reconstruction, got video_retriever",
            errors,
        )

    def test_success_requires_result(self):
        result = {
            "status": "success",
            "tool": "reconstruction",
        }

        valid, errors = validate_tool_result(result, "reconstruction")

        self.assertFalse(valid)
        self.assertIn("success result must contain result", errors)

    def test_error_requires_error_object(self):
        result = {
            "status": "error",
            "tool": "reconstruction",
        }

        valid, errors = validate_tool_result(result, "reconstruction")

        self.assertFalse(valid)
        self.assertIn("error result must contain an error object", errors)


class ToolExecutorTests(unittest.TestCase):
    def setUp(self):
        self.executor = ToolExecutor(
            tool_modes={
                "scene_conceiver": "mock",
                "reconstruction": "mock",
            }
        )

    @patch("agent.executor.call_tool")
    def test_executor_passes_through_valid_result(self, mock_call_tool):
        mock_call_tool.return_value = {
            "status": "success",
            "tool": "scene_conceiver",
            "result": {"scene_plan": "ok"},
        }

        result = self.executor.execute(
            "scene_conceiver",
            {"scene_requirement": {}},
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tool"], "scene_conceiver")
        self.assertEqual(result["result"], {"scene_plan": "ok"})
        mock_call_tool.assert_called_once_with(
            "scene_conceiver",
            {"scene_requirement": {}},
            mode="mock",
        )

    @patch("agent.executor.call_tool")
    def test_executor_normalizes_invalid_result(self, mock_call_tool):
        mock_call_tool.return_value = {
            "success": True,
            "data": {},
        }

        result = self.executor.execute(
            "reconstruction",
            {},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["tool"], "reconstruction")
        self.assertEqual(
            result["error"]["type"],
            "invalid_tool_result",
        )
        self.assertTrue(result["error"]["validation_errors"])

    @patch("agent.executor.call_tool")
    def test_executor_normalizes_tool_exception(self, mock_call_tool):
        mock_call_tool.side_effect = RuntimeError("backend failed")

        result = self.executor.execute(
            "reconstruction",
            {},
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["tool"], "reconstruction")
        self.assertEqual(
            result["error"]["type"],
            "execution_error",
        )
        self.assertEqual(
            result["error"]["message"],
            "backend failed",
        )


if __name__ == "__main__":
    unittest.main()
