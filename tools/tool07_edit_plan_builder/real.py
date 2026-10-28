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
    if input_data.get('backend') == 'ufo':
        from mcp_servers.ufo.planning.service import run as plan_ufo
        return plan_ufo(input_data)
    arguments = dict(input_data)
    backend = arguments.pop('backend', 'dggt')
    if backend != 'dggt':
        raise ValueError('backend must be ufo or dggt')
    return TOOL.execute(arguments)
