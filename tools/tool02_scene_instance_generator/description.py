"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='scene_instance_generator',
    description='From one verified failure analysis, write the single target scene and three implementation paths (vehicle edit, ego input edit, correct ego trajectory) with a matching base-video requirement each.',
    when_to_use='Turn a successful failure_analyzer result into a buildable scenario with retrievable base-video conditions.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'failure_analysis accepts the successful failure_analyzer result, envelope or absolute result_ref. It requires failure.ego_cause, attack, environment.time and verification.matched=true.',
             'output': 'Select one candidate. Pass base_video_requirement verbatim and the same candidate_id to video_retriever. After retrieval, pass the complete tool02 result or result_ref as scenario_result, the same candidate_id, and the tool03 result or result_ref as retrieval_result to edit_planner. target_scene and attack are shared by all candidates.'},
    rules=[
        'target_scene names only weather, time, road layout, ego maneuver and the other vehicle; no failure outcome or cause.',
        'attack repeats the measured tool01 structure unchanged and no candidate may replace it.',
        'base_video_requirement keeps what the edits cannot create and the vehicles or space the edits rely on; retrieval text and edit strategies must fit each other.',
        'Vehicles stay generic; the background stays sparse; no parameters, timestamps or trajectory points.',
        'ego_input_edit_strategy synchronizes ego speed/timing with the edited attack vehicle at the conflict. Preserve meaningful approach speed and task difficulty; avoid early stopping, crawling or making the conflict trivial. Use ego_correct_trajectory_strategy only for post-conflict safe behavior.',
    ],
)
