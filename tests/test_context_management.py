"""Check that context compression preserves evidence and exact artifact lineage."""
from copy import deepcopy
import json

from agent.state import AgentState
from agent.orchestrator import Orchestrator
from agent.tool_descriptions import get_tool_descriptions


def test_catalog_keeps_input_constraints_without_duplicate_output_schemas():
    full = get_tool_descriptions()
    compact = get_tool_descriptions(compact=True)
    assert [t["name"] for t in compact] == [t["name"] for t in full]
    for original, shortened in zip(full, compact):
        assert shortened["input_schema"]["required"] == original["input_schema"]["required"]
        assert shortened["input_schema"].get("$defs", {}).keys() == original["input_schema"].get("$defs", {}).keys()
        assert shortened["input_schema"]["properties"].keys() == original["input_schema"]["properties"].keys()
        assert "output_schema" not in shortened
        assert shortened["when_to_use"] == original["when_to_use"]
        assert shortened["handoff"] == original["handoff"]
        assert shortened["rules"] == original["rules"]
        assert set(shortened["outputs"]) == set(original["outputs"])
    assert len(json.dumps(compact)) < len(json.dumps(full)) * .70


def test_prompt_compression_is_bounded_and_keeps_recovery_and_contracts():
    planner = Orchestrator(llm_client=object())
    messages = planner._build_prompt("test", AgentState("test").to_orchestrator_dict())
    assert sum(len(m["content"]) for m in messages) < 60000
    assert "revise the earliest affected retrieval or editing step" in messages[0]["content"]
    schema = next(t for t in json.loads(messages[1]["content"])["available_tools"] if t["name"] == "attack_vehicle_editor")["input_schema"]
    assert schema["properties"]["operation"]["enum"] == ["edit_existing", "add_vehicle"]
    assert "VehicleAsset" in schema["$defs"]


def test_history_references_keep_exact_revision_not_latest_scene():
    state = AgentState("edit")
    first = {"scene_ref": {"scene_id": "first"}}
    second = {"scene_ref": {"scene_id": "second"}}
    for payload, path in [(first, "/first.json"), (second, "/second.json")]:
        state.add_tool_result("video_retriever", {"status": "success", "tool": "video_retriever", "result": payload}, path)
    state.add_decision(3, {"tool_name": "attack_vehicle_editor", "arguments": {"scene_ref": first["scene_ref"]}})
    state.add_decision(4, {"tool_name": "attack_vehicle_editor", "arguments": {"scene_ref": {"scene_id": "external"}}})
    snapshot = deepcopy(state.to_dict())
    context = state.to_orchestrator_dict()
    assert context["history"][2]["decision"]["arguments"]["scene_ref"] == "/first.json"
    assert context["history"][3]["decision"]["arguments"]["scene_ref"] == {"scene_id": "external"}
    assert state.to_dict() == snapshot


def test_full_logs_remain_intact_and_failed_feedback_stays_visible():
    state = AgentState("edit")
    attack = {"source_scene_ref": {"scene_id": "first"}, "target_track": "v1"}
    state.add_tool_result("attack_vehicle_editor", {"tool": "attack_vehicle_editor", "status": "success", "result": attack}, "/attack.json")
    state.add_decision(2, {"tool_name": "ego_trajectory_editor", "arguments": {"attack_result": deepcopy(attack)}})
    error = {"tool": "ego_trajectory_editor", "status": "error", "error": {"type": "timeout", "message": "retry once"}}
    state.add_tool_result("ego_trajectory_editor", error, "/error.json")
    original = deepcopy(state.to_dict())
    context = state.to_orchestrator_dict()
    assert context["history"][1]["decision"]["arguments"]["attack_result"] == "/attack.json"
    assert "source_scene_ref" not in context["history"][0]["result"]["result"]
    assert context["history"][-1]["result"] == error
    assert state.to_dict() == original
