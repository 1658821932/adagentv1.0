"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='scene_validator',
    description='Check edited-render requirement consistency and visible geometry, motion and traffic plausibility.',
    when_to_use='Check a new or revised edited render.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'Use an actual path from editing result.rendered_videos and the original scene_requirement; original recordings are not edited renders.', 'output': 'Continue only when result.passed=true. Otherwise use issues and frame evidence to revise the affected retrieval/editing stage.'},
    rules=[
        'The configured VLM evaluates sampled frames unless an explicit local evaluation is supplied. File existence alone does not validate the scene.',
        'Visual plausibility does not establish policy behavior, exact metric distances or dynamics guarantees.',
    ],
)
