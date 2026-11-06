"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='failure_analyzer',
    description='Analyze one WorldEngine scene from official metrics, images and numeric evidence.',
    when_to_use='Diagnose a scene with archived evidence; reuse saved analysis when sufficient.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'Use the existing scene token and analysis configuration.', 'output': 'Pass result.failure_analysis to scene_instance_generator.failure_analysis; report_path/result_path locate the full evidence.'},
    rules=[
        'This tool analyzes one scene; it does not summarize a free-form list of failure cases.',
    ],
)
