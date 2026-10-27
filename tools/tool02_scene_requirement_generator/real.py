"""LLM-backed scenario requirements using the shared typed runtime."""
from pathlib import Path
from tools.llm_runtime import LLMTool
from .schemas import Input, Response, Output

TOOL = LLMTool(
    name="scene_requirement_generator",
    directory=Path(__file__).resolve().parent,
    input_model=Input, response_model=Response, output_model=Output,
    payload_builder=lambda data: data.failure_analysis,
    enrich_result=lambda data, result: {**result, "source_analysis": data.failure_analysis},
)


def run(input_data: dict) -> dict:
    return TOOL.execute(input_data)
