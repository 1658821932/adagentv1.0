"""Reference-passing handoff: advertised schemas and tool04 teaching errors."""
import json
import unittest


class ReferencePassingSchemaTests(unittest.TestCase):
    def _advertised(self, tool_name):
        from agent.tool_descriptions import get_function_tools
        for fn in get_function_tools():
            if fn["function"]["name"] == tool_name:
                return fn["function"]["parameters"]["properties"]["arguments"]
        raise AssertionError(f"tool not advertised: {tool_name}")

    def test_handoff_fields_are_string_references(self):
        cases = {
            "edit_planner": "retrieval_result",
            "ego_trajectory_editor": "attack_result",
        }
        for tool, field in cases.items():
            schema = self._advertised(tool)["properties"][field]
            self.assertEqual(schema.get("type"), "string", f"{tool}.{field} must be advertised as a reference string")
            self.assertIn("result_ref", schema.get("description", ""))

    def test_attack_supports_task_bound_scene_and_result_reference(self):
        schema = self._advertised("attack_vehicle_editor")["properties"]["scene_ref"]
        variants = schema["anyOf"]
        self.assertTrue(any(v.get("type") == "string" for v in variants))
        self.assertTrue(any(v.get("$ref", "").endswith("/SceneReference") for v in variants))
        self.assertIn("directly by the task", schema["description"])
        from mcp_servers.scene_editing.adapter.schemas import AttackInput
        scene = dict(scene_id="provided",scene_token="token",sample_root="/scene",dataset="nuplan",
            matched_camera="CAM_F0",frame_count=24,manifest_path="/scene/map.json",
            frames_path="/scene/frames.json",cameras_path="/scene/cameras.json",cameras={},images_complete=True)
        validated = AttackInput.model_validate(dict(scene_ref=scene,instruction="Slow the lead car",operation="edit_existing"))
        self.assertEqual(validated.scene_ref.scene_id,"provided")

    def test_non_handoff_tools_keep_object_schemas(self):
        schema = self._advertised("scene_instance_generator")["properties"]["failure_analysis"]
        self.assertNotEqual(schema.get("type"), "string")


class MockFieldHidingTests(unittest.TestCase):
    def test_mock_only_fields_not_advertised(self):
        from agent.tool_descriptions import get_function_tools
        for fn in get_function_tools():
            props = fn["function"]["parameters"]["properties"].get("arguments", {}).get("properties", {})
            for field, prop in props.items():
                self.assertFalse(prop.get("description", "").startswith("Mock-only"),
                                 f"{fn['function']['name']}.{field} is mock-only and must not be advertised")

    def test_video_retriever_mock_field_hidden(self):
        from agent.tool_descriptions import get_function_tools
        for fn in get_function_tools():
            if fn["function"]["name"] == "video_retriever":
                props = fn["function"]["parameters"]["properties"]["arguments"]["properties"]
                self.assertNotIn("mock_video_path", props)


class NormalizeRetrievalTests(unittest.TestCase):
    def test_hand_built_dict_with_dead_paths_gets_teaching_error(self):
        from tools.tool04_edit_planner.schemas import normalize_retrieval
        value = {"scene_ref": {"scene_token": "x", "frames_path": "/no/such/frames",
                               "cameras_path": "/no/such/cameras", "scenario_pkl": "/no/such.pkl"}}
        with self.assertRaisesRegex(ValueError, "result_ref"):
            normalize_retrieval(value)

    def test_envelope_dict_is_trusted(self):
        from tools.tool04_edit_planner.schemas import normalize_retrieval
        envelope = {"tool": "video_retriever", "status": "success",
                    "result": {"scene_ref": {"scene_token": "x", "frames_path": "/fake/frames.json"}}}
        self.assertEqual(normalize_retrieval(envelope)["scene_ref"]["scene_token"], "x")


if __name__ == "__main__":
    unittest.main()
