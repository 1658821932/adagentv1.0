"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='video_retriever',
    description='Retrieve a locally indexed base video, its bound multi-camera scene, and the next two distinct ranked scenes for fallback.',
    when_to_use='Find a source scene for the selected tool02 candidate.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'Pass the selected candidate.retrieval_query verbatim as requirement, with the same candidate_id. Never pass base_video_requirement or add conditions assigned to later editing.',
             'output': 'Pass the successful result_ref (preferred), envelope or complete result to edit_planner.retrieval_result. Tool04 automatically tries alternatives ranks 2 and 3 when the winner is unsuitable.'},
    rules=[
        'The query is compact: match environment, road form, ego direction and required clear space; do not require planned edits, other-vehicle motion, conflict or policy failure.',
        'Local index only. match_score measures relevance; tool04 checks structural suitability.',
        'scene_ref is the full bound scene; source_segment is the index-match range; video_path is the exported clip, not a synchronized multi-camera crop. alternatives contain distinct ranks 2 and 3.',
    ],
)
