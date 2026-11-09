from tools.llm_runtime import make_description
from .schemas import Input, Output
DESCRIPTION=make_description(name='edit_planner',
    description='Review a retrieved base scene and produce a coarse plan plus validated attack-editor arguments.',
    when_to_use='Between tool03 retrieval and tool05 attack editing.',input_model=Input,output_model=Output,
    handoff={'input':'retrieval_result accepts the successful tool03 envelope, result or history result_ref; also supply tool02 scene_requirement, target_effect and the selected edit_strategy. Optional vehicle supplies a real added-car asset and dimensions.',
             'output':'ready: pass result.attack_input unchanged as attack_vehicle_editor arguments. Keep target_effect for tool06. unsuitable: retrieve again using retrieval_suggestion. needs_information: obtain the named resources, then review again.'},
    rules=['Initial observation is 12 CAM_F0 frames in three chronological 2x2 contact sheets. The VLM may request other available cameras internally, up to two cameras per round and two extra rounds.',
           'This quick review does not run trajectory optimization, PDM or scene rendering.',
           'Existing targets must be valid vehicles observed throughout the scene. Added-car assets and dimensions come from supplied input, not model estimates.',
           'Non-ready outputs have attack_input=null; a coarse plan alone is not an executable attack request.'])
