"""Compact tool guide derived from executable schemas."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name="scene_instance_generator",
    description="Expand one failure category into one new scene requirement, returned as four fields: scene_requirement, required_conditions, required_actors, target_effect.",
    when_to_use="Pass saved failure_analysis; variation_hints and existing_instances are optional. Forward the successful output to tool03 as scene_requirement. Source analysis and hints remain in local run records referenced by meta.record_dir.",
    input_model=Input, output_model=Output,
)
