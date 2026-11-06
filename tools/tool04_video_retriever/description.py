"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='video_retriever',
    description='Retrieve a local camera video and its bound annotated multi-camera scene.',
    when_to_use='Retrieve a source scene for editing or downstream use.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'Use the selected scene_instance_generator candidate.base_video_requirement as requirement.', 'output': 'Pass the successful history entry.result_ref or result.scene_ref to attack_vehicle_editor.scene_ref. source_file is the indexed original; video_path is the exported search clip.'},
    rules=[
        'No internet search. Real mode requires a valid scene binding.',
        'scene_ref describes the full source scene. source_segment is the matched index range, not a synchronized multi-camera crop.',
        'Check returned asset/image availability before requesting an operation that needs those files.',
    ],
)
