"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='failure_analyzer',
    description='Analyze one WorldEngine scene from official metrics, images and numeric evidence.',
    when_to_use='Diagnose one closed-loop scene from its archived official metrics and recorded images.',
    input_model=Input,
    output_model=Output,
    handoff={
        'input': 'State BOTH the simulation record and the scene token: token alone does not say which closed-loop run produced the official metrics.',
        'output': 'Pass result.failure_analysis to scene_instance_generator.failure_analysis; result.simulation records which run produced it.',
    },
    rules=[
        'This tool analyzes one scene; it does not summarize a free-form list of failure cases.',
        'A scene can be simulated more than once. simulation selects the closed-loop run '
        '("<experiment>/<batch>", a unique batch name, or an archive directory); token selects '
        'the scene inside it. Metrics and diagnosis differ between runs. Map user wording: '
        '仿真号/simulation ID -> simulation, 场景号/scene ID -> token.',
        'The token is a WorldEngine closed-loop scene identity obtained from policy_runner -> '
        'failure_case_filter or supplied by the user. Short and full identifiers both work. '
        'Do not pass a retrieval scene_ref token.',
        'The token must belong to the chosen simulation archive; otherwise the call fails with '
        'the archive scene count.',
        'One scene takes tens of seconds to a few minutes and every call runs fresh, with no '
        'cached reuse. Do not duplicate the call while it is running.',
        'If the analysis reports missing archived data or an unusable archive, fix the input or '
        'choose another evidenced token; never substitute a guessed failure type.',
    ],
)
