"""Rule-based scene requirement compiler."""
from __future__ import annotations
from typing import Any
from tools.llm_json import ask_json

def _unwrap(value: Any) -> Any:
    if isinstance(value, dict) and "result" in value: value = value["result"]
    if isinstance(value, dict) and "failure_analysis" in value: value = value["failure_analysis"]
    return value

def _items(value: Any) -> list[str]:
    if isinstance(value, str): return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)): return [str(x).strip() for x in value if str(x).strip()]
    return []

def run(input_data: dict) -> dict:
    analysis = _unwrap(input_data.get("failure_analysis"))
    if not isinstance(analysis, dict): raise ValueError("scene_requirement_generator requires a failure_analysis object")
    result = ask_json(input_data, """You generate autonomous-driving scenario requirements. Return ONLY JSON with keys: scene_requirement (string), failure_type (string), failure_reason (string), required_conditions (array of strings), required_actors (array of strings), risk_factors (array of strings), target_effect (string), hazard_summary (string). Ground every field in the supplied failure analysis; do not invent unavailable video facts.""", analysis)
    required = ("scene_requirement", "required_conditions", "required_actors", "risk_factors", "target_effect")
    missing = [key for key in required if key not in result]
    if missing: raise ValueError("LLM scene requirement is missing: " + ", ".join(missing))
    return {"status":"success", "tool":"scene_requirement_generator", "result": {
        **result, "source_analysis": analysis}}
