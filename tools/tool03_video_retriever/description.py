"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='video_retriever',
    description='Retrieve a local camera video and its bound annotated multi-camera scene.',
    when_to_use='Retrieve a source scene for editing or downstream use.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'Use the selected scene_instance_generator candidate.base_video_requirement as requirement.', 'output': 'Pass the successful history entry.result_ref, envelope or result to edit_planner.retrieval_result. source_file is the indexed original; video_path is the exported search clip.'},
    rules=[
        'For scenario editing, pass the complete retrieval result or result_ref to edit_planner.retrieval_result before expensive attack search.',
        'No internet search. Real mode requires a valid scene binding.',
        'scene_ref describes the full source scene. source_segment is the matched index range, not a synchronized multi-camera crop.',
        'Check returned asset/image availability before requesting an operation that needs those files.',
    ],
)
