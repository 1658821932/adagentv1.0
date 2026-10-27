"""Orchestrator-facing guide; input/output fields are generated from schemas."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name="scene_requirement_generator",
    description="Use an LLM to generate targeted scenario requirements from failure evidence.",
    when_to_use="Use after failure analysis, or when supplied analysis needs to become generation requirements. Proposed targets are not observed facts.",
    input_model=Input, output_model=Output,
)
