"""Mock review: reports missing information instead of claiming a scene check."""
from .schemas import Input, Output
from tools.llm_runtime import validation_details


def run(input_data):
    try:
        data = Input.model_validate(input_data)
    except (ValueError, TypeError, KeyError) as error:
        return dict(status="error", tool="edit_planner", error=dict(
            type="input_validation_error", message="Tool input does not match its schema",
            validation_errors=validation_details(error)))
    result = Output.model_validate(dict(
        suitability="needs_information",
        reasons=["Mock only: road, lanes and space were not inspected."],
        checks=dict(road="unknown", lanes="unknown", space="unknown"),
        adapted_plan=None, retrieval_suggestion=None,
        scene_id=data.scene_ref.scene_id, scene_token=data.scene_ref.scene_token,
        candidate_id=data.candidate_id,
        target_scene=data.scenario.target_scene,
        attack=data.scenario.attack.model_dump(mode="json"),
        evidence={"mode": "mock"}))
    return dict(status="success", tool="edit_planner", result=result.model_dump(mode="json"),
                meta={"mode": "mock"})
