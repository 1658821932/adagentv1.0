"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='scene_instance_generator',
    description='From one verified failure analysis, write the single target scene and three implementation paths (vehicle edit, ego input edit, correct ego trajectory), each with source-video conditions and a compact retrieval query.',
    when_to_use='Turn a successful failure_analyzer result into a buildable scenario with retrievable base-video conditions.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'failure_analysis accepts the successful failure_analyzer result, envelope or absolute result_ref. It requires violation_step, failure.ego_cause, the final attack, environment.time and the visual verification record.',
             'output': 'Select one candidate. Pass its retrieval_query verbatim as video_retriever.requirement with the same candidate_id. After retrieval, pass the complete tool02 result or result_ref as scenario_result, the same candidate_id, and the tool03 result or result_ref as retrieval_result to edit_planner. target_scene and attack are shared by all candidates.'},
    rules=[
        'target_scene names only weather, time, road layout, ego maneuver and the other vehicle; no failure outcome or cause.',
        'attack repeats the measured tool01 structure unchanged and no candidate may replace it.',
        'ego_action and other_action describe direction only. Lane position is represented separately by ego_lane_keeping and other_lane_keeping; speed changes never become actions.',
        'base_video_requirement keeps what the edits cannot create and the vehicle or clear space the edits rely on. Prefer adding a vehicle into clear space.',
        'retrieval_query is a separate compact search phrase, at most 48 characters. It names only environment, road form, ego direction and required clear space; never other-vehicle action, lane offset, speed, timing, conflict or editing.',
        'Vehicles stay generic; the background stays sparse; no parameters, timestamps or trajectory points.',
        'ego_input_edit_strategy synchronizes ego speed/timing with the edited attack vehicle at the conflict. Preserve meaningful approach speed and task difficulty; avoid early stopping, crawling or making the conflict trivial. Use ego_correct_trajectory_strategy only for post-conflict safe behavior.',
    ],
)
