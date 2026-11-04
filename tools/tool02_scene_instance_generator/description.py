"""Merged scenario conception and candidate retrieval guide."""
from tools.llm_runtime import make_description
from .schemas import Input, Output
DESCRIPTION = make_description(
    name="scene_instance_generator",
    description="Expand a failure category into one target scene, ego's expected behavior, and three alternative base-video/edit-strategy pairs. Includes the former tool03 conception responsibility.",
    when_to_use="Use saved failure_analysis. Select one candidate and pass candidates[i].base_video_requirement unchanged as video_retriever.requirement. Retain that candidate's edit_strategy for downstream editing. Empty base scenes are valid for adding a vehicle. Prefer editing one other vehicle and optionally ego trajectory.",
    input_model=Input, output_model=Output,
)
