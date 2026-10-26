"""LLM-backed construction of a concrete natural-language edit plan."""
from __future__ import annotations
from typing import Any
from tools.llm_json import ask_json

def run(input_data: dict[str, Any]) -> dict[str, Any]:
    request = input_data.get("edit_request") or input_data.get("edit_ideas")
    if not request:
        raise ValueError("edit_plan_builder requires edit_request or edit_ideas")
    result = ask_json(input_data, """You construct a precise autonomous-driving scene edit plan for a downstream Smart Edit configuration generator. Return ONLY JSON with keys: edit_plan (string), target_instances (array), operations (array of objects with operation, target, parameters), constraints (array of strings). Do not generate Smart Edit JSON and do not invent instance IDs or coordinates absent from the input.""", {"request": request, "scene_info": input_data.get("scene_info"), "instances": input_data.get("instances")})
    for key in ("edit_plan", "operations"):
        if key not in result:
            raise ValueError(f"LLM edit plan is missing: {key}")
    return {"status": "success", "tool": "edit_plan_builder", "result": result}
