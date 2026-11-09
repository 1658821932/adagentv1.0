"""Mock coarse planning does not claim to inspect actual scene assets."""
from .schemas import Input, Output
from tools.llm_runtime import validation_details

def run(input_data):
    try:
        data=Input.model_validate(input_data)
    except (ValueError,TypeError,KeyError) as error:
        return dict(status='error',tool='edit_planner',error=dict(type='input_validation_error',
                    message='Tool input does not match its schema',validation_errors=validation_details(error)))
    result=Output(suitability='needs_information',reasons=['Mock only: no real scene inspection or planning was performed.'],
        scene_id=data.scene_ref.scene_id,scene_token=data.scene_ref.scene_token,target_effect=data.target_effect,
        evidence={'mode':'mock'},edit_plan=None,retrieval_suggestion=None)
    return dict(status='success',tool='edit_planner',result=result.model_dump(),meta={'mode':'mock'})
