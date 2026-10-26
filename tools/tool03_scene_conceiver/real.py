"""Deterministic scenario conception from a scene requirement contract."""
from __future__ import annotations
from typing import Any
from tools.llm_json import ask_json

def _unwrap(value: Any) -> Any:
    return value.get("result") if isinstance(value, dict) and "result" in value else value
def _items(value: Any) -> list[str]:
    if isinstance(value, str): return [value.strip()] if value.strip() else []
    return [str(x).strip() for x in value if str(x).strip()] if isinstance(value, list) else []

def run(input_data: dict) -> dict:
    requirement = _unwrap(input_data.get("scene_requirement"))
    if not isinstance(requirement, dict): raise ValueError("scene_conceiver requires a scene_requirement object")
    result = ask_json(input_data, """You design a realistic autonomous-driving scenario from requirements. Return ONLY JSON with keys: scene_plan (string), base_video_requirement (string), required_conditions (array), required_actors (array), edit_ideas (array of objects with operation, target_role, parameters), expected_effect (string), retrieval_query (string). Keep edits physically plausible and tied to the requirements.""", requirement)
    for key in ("scene_plan", "base_video_requirement", "edit_ideas", "retrieval_query"):
        if key not in result: raise ValueError(f"LLM scene conception is missing: {key}")
    return {"status":"success", "tool":"scene_conceiver", "result": {
        **result, "source_requirement": requirement}}
