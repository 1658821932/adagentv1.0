from tools.llm_runtime import make_description
from .schemas import Input, Output
DESCRIPTION=make_description(name='edit_planner',
    description='Review a retrieved base scene and produce 1..3 unranked edit alternatives plus an auto-selection attack-editor handoff.',
    when_to_use='Between tool03 retrieval and tool05 attack editing.',input_model=Input,output_model=Output,
    handoff={'input':'retrieval_result accepts the successful tool03 envelope, result or history result_ref; also supply tool02 scene_requirement, target_effect and the selected edit_strategy. Optional vehicle supplies a real added-car asset and dimensions.',
             'output':'ready: edit_options are suggestions for the attack SubAgent; pass result.attack_input (operation=auto, plan_options, optional asset) unchanged as attack_vehicle_editor arguments. Keep target_effect for tool06. unsuitable: retrieve again using retrieval_suggestion. needs_information: obtain the named resources, then review again.'},
    rules=['Initial observation includes 12 CAM_F0 frames in three chronological 2x2 sheets and, when source transforms are available, an original-scene BEV with the recorded ego route. Internally the VLM may query a catalog vehicle or ego trajectory at most three times (24 sampled observations each), or request other cameras for two rounds of up to two cameras.',
           'This quick review does not run trajectory optimization, PDM, policy simulation or photorealistic scene rendering. Original future motion does not veto planning. Recorded proximity is advisory; detailed motion checks belong to tool05.',
           'Existing targets must be valid vehicles observed throughout the scene. Added-car assets and dimensions come from supplied input, not model estimates.',
           'edit_existing starts from the recorded t=0 pose by default; tool05 may create a bounded initial-position/yaw-adjusted candidate version with full rechecking. add_vehicle may choose a supported new initial pose and stationary behavior. Presence of a real actor alone does not require choosing it over an added vehicle.',
           'Non-ready outputs have attack_input=null; a coarse plan alone is not an executable attack request.',
           'Canonical added-vehicle asset: /inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagent/data/assets/vehicle_01.ply with size_m [4.432, 2.033, 1.637] (length, width, height; see data/assets/manifest.json). When the caller supplies it, add_vehicle is available. Describe a proposed mapped turn or other movement in the option instruction.'])
