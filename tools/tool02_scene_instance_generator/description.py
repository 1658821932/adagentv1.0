"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='scene_instance_generator',
    description='Generate one target scene, one global ego goal and three source-video/edit-strategy candidates.',
    when_to_use='Construct a new scenario from saved failure analysis.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'failure_analysis: saved category or successful failure_analyzer result.', 'output': 'Choose candidates[i]; pass base_video_requirement verbatim to video_retriever.requirement and retain its paired edit_strategy. target_effect applies to the whole target scene.'},
    rules=[
        'Candidate requirements describe source content to retrieve, not the final edited hazard.',
        'Planning supports adding, deleting or modifying vehicles and their trajectories, plus ego trajectory edits. Prefer changing at most one other vehicle; execution still follows the chosen editor contract.',
    ],
)
