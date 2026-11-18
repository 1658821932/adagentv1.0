"""Compact scenario requirements, with source inputs retained in local run records."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from tools.llm_runtime import LLMTool
from mcp_servers.runtime import write_json
from .schemas import Input, Response, Output, normalized

RECORDS_ROOT = Path(__file__).resolve().parents[2] / "logs" / "tool02_scene_instance_generator"


def _validate_result(data, output):
    upstream = data.failure_analysis.get("attack") or {}
    if output.attack.model_dump(mode="json") != upstream:
        raise ValueError("attack must repeat the upstream tool01 structure unchanged")
    # The construction plan is base video + the two edits; the correct trajectory is
    # how the ego reacts, so rewording it alone must not look like a new plan.
    combinations = {
        (normalized(item.base_video_requirement),
         normalized(item.attack_edit_strategy or "null"),
         normalized(item.ego_input_edit_strategy or "null"))
        for item in output.candidates
    }
    if len(combinations) != 3:
        raise ValueError("Provide three distinct base-video and editing-strategy alternatives")


def _payload(data):
    """Only failure, attack, environment and verification reach the model; source stays local."""
    payload = data.model_dump(mode="json")
    analysis = payload.get("failure_analysis")
    if isinstance(analysis, dict):
        analysis.pop("source", None)
    return {key: value for key, value in payload.items() if value not in ({}, [])}


TOOL = LLMTool(
    name="scene_instance_generator", directory=Path(__file__).resolve().parent,
    input_model=Input, response_model=Response, output_model=Output,
    payload_builder=_payload,
    validate_result=_validate_result,
)


def execute(input_data, *, mock_result=None):
    # Source analysis/hints belong in records, not in the scenario result.
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
