"""Orchestrator guide generated from executable field definitions."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name="scene_conceiver",
    description="Use an LLM to plan a scenario, source-video requirements and feasible editing ideas.",
    when_to_use="Use after requirements are available and before video retrieval. Pass retrieval_query or base_video_requirement to video_retriever.requirement.",
    input_model=Input, output_model=Output,
)
