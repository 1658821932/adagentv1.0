"""LLM scenario conception using the shared typed runtime."""
from pathlib import Path
from tools.llm_runtime import LLMTool
from .schemas import Input, Response, Output


def _enrich(data, result):
    return {
        **result,
        "source_requirement": data.scene_requirement,
    }


TOOL = LLMTool(
    name="scene_conceiver", directory=Path(__file__).resolve().parent,
    input_model=Input, response_model=Response, output_model=Output,
    payload_builder=lambda data: data.scene_requirement,
    enrich_result=_enrich,
)


def run(input_data: dict) -> dict:
    return TOOL.execute(input_data)
