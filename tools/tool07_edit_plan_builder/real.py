"""LLM editing plans with typed output and instance grounding checks."""
from pathlib import Path
from tools.llm_runtime import LLMTool
from .schemas import Input, Response, Output, validate_grounding

TOOL = LLMTool(
    name="edit_plan_builder", directory=Path(__file__).resolve().parent,
    input_model=Input, response_model=Response, output_model=Output,
    validate_result=validate_grounding,
)


def run(input_data: dict) -> dict:
    return TOOL.execute(input_data)
