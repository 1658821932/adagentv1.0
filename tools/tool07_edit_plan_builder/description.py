"""Orchestrator-facing guide generated from the editing contracts."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name="edit_plan_builder",
    description="Use an LLM to turn an edit request and discovered scene context into a grounded natural-language editing plan.",
    when_to_use="For UFO use backend=ufo and the backend_contracts.ufo schema; for DGGT use the base schema. Use after reconstruction and instance discovery when an edit must be specified. Supply actual instances so IDs can be checked. Existing trajectory edits are relative to original per-frame motion; cut-in timing comes from the current task.",
    input_model=Input, output_model=Output,
)

from mcp_servers.ufo.planning.service import Input as UFOInput, Output as UFOOutput
UFO_DESCRIPTION = make_description(name='edit_plan_builder', description='Typed UFO visual/metric planning.',
    when_to_use='For backend=ufo after calibration and/or target selection.', input_model=UFOInput, output_model=UFOOutput)
DESCRIPTION['backend_contracts'] = {'ufo': UFO_DESCRIPTION}
