"""Compact scenario requirements, with source inputs retained in local run records."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from tools.llm_runtime import LLMTool
from mcp_servers.runtime import write_json
from .schemas import Input, Response, Output, normalized

RECORDS_ROOT = Path(__file__).resolve().parents[2] / "logs" / "tool02_scene_instance_generator"


def _validate_result(data, output):
    if normalized(output.scene_requirement) in {normalized(item) for item in data.existing_instances}:
        raise ValueError("Scene requirement duplicates an existing description; propose a different interaction or context")
    pairs = {(normalized(item.base_video_requirement), normalized(item.edit_strategy)) for item in output.candidates}
    if len(pairs) != 3:
        raise ValueError("Provide three distinct base-video and editing-strategy alternatives")


TOOL = LLMTool(
    name="scene_instance_generator", directory=Path(__file__).resolve().parent,
    input_model=Input, response_model=Response, output_model=Output,
    payload_builder=lambda data: data.model_dump(mode="json"),
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
