"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='video_retriever',
    description='Retrieve a locally indexed base video and its bound multi-camera scene.',
    when_to_use='Find a source scene for the selected tool02 candidate.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'Pass the selected candidate.base_video_requirement unchanged as requirement, with the same candidate_id. Do not add conditions assigned to later editing.',
             'output': 'Pass the successful result_ref (preferred), envelope or complete result to edit_planner.retrieval_result.'},
    rules=[
        'Match only the requested source-video conditions; do not require planned edits, conflict or policy failure.',
        'Local index only. match_score measures relevance; tool04 checks structural suitability.',
        'scene_ref is the full bound scene; source_segment is the index-match range; video_path is the exported clip, not a synchronized multi-camera crop.',
    ],
)
