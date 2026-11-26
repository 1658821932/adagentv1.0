"""Tool 02: scenario candidate generator with per-candidate asset requirements and add-vehicle priority."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from mcp_servers.runtime import write_json
from tools.llm_runtime import LLMTool
from .schemas import Input, Output, Response, normalized

RECORDS_ROOT = Path(__file__).resolve().parents[2] / "logs" / "tool02_scene_instance_generator"


def _has_target_vehicle(data: Input) -> bool:
    driver = data.failure_analysis["failure"]["failure_driver"]
    return driver in ("collision_with_vehicle", "departure_by_other")


def _enrich_result(data: Input, result: dict) -> dict:
    failure = data.failure_analysis["failure"]
    kinematics = data.failure_analysis["kinematics"]
    other_size = (kinematics.get("other") or {}).get("size_m")

    enriched_candidates = []
    for item in result.get("candidates") or []:
        cand = dict(item)
        asset = cand.get("asset_requirement")
        if asset is not None:
            if not other_size:
                raise ValueError("cannot build asset_requirement without upstream kinematics.other.size_m")
            cand["asset_requirement"] = {
                "vehicle_class": asset["vehicle_class"],
                "size_m": [float(x) for x in other_size],
            }
        enriched_candidates.append(cand)

    source_bev = (data.failure_analysis.get("source") or {}).get("bev_image")
    return {
        "target_scene": result["target_scene"],
        "failure_context": {
            "outcome": failure["outcome"],
            "failure_driver": failure["failure_driver"],
            "map_context": failure["map_context"],
        },
        "kinematics": dict(kinematics),
        "source_bev_image": source_bev,
        "candidates": enriched_candidates,
    }


def _validate_result(data: Input, output: Output) -> None:
    has_target = _has_target_vehicle(data)
    if not has_target:
        for item in output.candidates:
            if item.attack_edit_strategy is not None or item.asset_requirement is not None:
                raise ValueError(
                    "single-vehicle/non-vehicle failure must have null attack_edit_strategy and null asset_requirement"
                )
    else:
        if output.candidates[0].asset_requirement is None:
            raise ValueError("cand_1 must prioritize adding a vehicle (asset_requirement cannot be null)")
        for item in output.candidates:
            if item.attack_edit_strategy is None:
                raise ValueError("vehicle-driven failure requires non-empty attack_edit_strategy on every candidate")

    combinations = {
        (
            normalized(item.base_video_requirement),
            normalized(item.attack_edit_strategy or "null"),
            normalized(item.ego_input_edit_strategy or "null"),
            "add" if item.asset_requirement is not None else "no_add",
        )
        for item in output.candidates
    }
    if len(combinations) != 3:
        raise ValueError("Provide three distinct base-video, asset-mode, and editing-strategy alternatives")


def _payload(data: Input) -> dict:
    """Send measured failure, kinematics summary, and environment to LLM; strip local paths."""
    analysis = dict(data.failure_analysis)
    analysis.pop("source", None)
    kin = dict(analysis.get("kinematics") or {})
    kin.pop("trajectory_ref", None)
    analysis["kinematics"] = kin
    failure = dict(analysis.get("failure") or {})
    failure.pop("target_track", None)
    analysis["failure"] = failure
    return {"failure_analysis": analysis}


TOOL = LLMTool(
    name="scene_instance_generator",
    directory=Path(__file__).resolve().parent,
    input_model=Input,
    response_model=Response,
    output_model=Output,
    payload_builder=_payload,
    enrich_result=_enrich_result,
    validate_result=_validate_result,
)


def execute(input_data, *, mock_result=None):
    try:
        data = Input.model_validate({k: v for k, v in input_data.items() if k != "_llm_client"})
    except (AttributeError, ValueError, TypeError):
        return TOOL.execute(input_data, mock_result=mock_result)
    run_dir = RECORDS_ROOT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "input.json", data.model_dump(mode="json"))
    result = TOOL.execute(input_data, mock_result=mock_result)
    result.setdefault("meta", {})["record_dir"] = str(run_dir)
    write_json(run_dir / "output.json", result)
    return result


def run(input_data: dict) -> dict:
    return execute(input_data)
