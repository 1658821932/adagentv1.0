"""Deterministic scenario plan using the real tool's contracts."""
from .real import TOOL


def _result(data):
    requirement=data.scene_requirement
    if isinstance(requirement, dict):
        summary=str(requirement.get("scene_requirement") or requirement.get("target_effect") or "requested driving scenario")
        conditions=requirement.get("required_conditions", [])
        actors=requirement.get("required_actors", [])
    else:
        summary=requirement
        conditions,actors=[],[]
    return {
        "scene_plan": "Prepare a source scene and edit it to satisfy: " + summary,
        "base_video_requirement": "Forward-facing driving video suitable for: " + summary,
        "required_conditions": conditions,
        "required_actors": actors,
        "edit_ideas": [{"operation":"modify_trajectory","target_role":"relevant traffic participant","parameters":{}}],
        "expected_effect": summary,
        "retrieval_query": "Forward-facing driving scene: " + summary,
    }


def run(input_data: dict) -> dict:
    return TOOL.execute(input_data, mock_result=_result)
