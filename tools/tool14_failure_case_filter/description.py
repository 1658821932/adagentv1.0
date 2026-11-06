"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='failure_case_filter',
    description='Select zero-score failure scenes from official closed-loop PDM results.',
    when_to_use="Filter a completed run's official PDM results.",
    input_model=Input,
    output_model=Output,
    handoff={'input': 'Use policy_runner.result_dir and the scenario_pkl corresponding to that execution; scene_token optionally selects one scene.', 'output': 'failure_cases contains selected scene identities and PDM scores; pass their tokens to failure_analyzer when analysis is needed.'},
    rules=[
        'Only official PDM total score == 0 selects a failure. Missing scores or ambiguous scene matches are errors; do not substitute geometry checks.',
    ],
)
