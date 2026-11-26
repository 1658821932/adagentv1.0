"""Deterministic mock of Tool 04 (edit_planner)."""
from .schemas import Input, Output


def run(input_data: dict) -> dict:
    data = Input.model_validate(input_data)
    cand = data.candidate
    fc = data.scenario_result.failure_context
    single = fc.failure_driver in ("departure_ego_only", "collision_non_vehicle")
    attack_input = None
    if not single:
        if cand.asset_requirement is not None:
            attack_input = {
                "scene_ref": data.scene_ref.model_dump(mode="json"),
                "instruction": f"{data.scenario_result.target_scene} {cand.attack_edit_strategy}",
                "operation": "add_vehicle",
                "maneuver": "moving",
                "vehicle": {"asset_path": "mock://assets/vehicle_01.ply", "size_m": [4.432, 2.033, 1.637]},
                "prepare_ego_history": cand.ego_input_edit_strategy is not None,
                "constraints": {"no_ego_contact_until_s": 0.0, "target_window_s": None},
                "reference_kinematics": data.scenario_result.kinematics,
                "reference_bev_image": data.scenario_result.source_bev_image,
            }
        else:
            attack_input = {
                "scene_ref": data.scene_ref.model_dump(mode="json"),
                "instruction": f"{data.scenario_result.target_scene} {cand.attack_edit_strategy}",
                "operation": "edit_existing",
                "target_track": "mock_track_01",
                "prepare_ego_history": cand.ego_input_edit_strategy is not None,
                "constraints": {"no_ego_contact_until_s": 0.0, "target_window_s": None},
                "reference_kinematics": data.scenario_result.kinematics,
                "reference_bev_image": data.scenario_result.source_bev_image,
            }
    result = Output.model_validate({
        "suitability": "ready",
        "reasons": ["Mock dual-BEV review passed road, lanes, and space checks."],
        "checks": {"road": "pass", "lanes": "pass", "space": "pass"},
        "target_window_s": None if single else [3.0, 5.5],
        "adapted_plan": {
            "attack_edit_strategy": None if single else cand.attack_edit_strategy,
            "ego_input_edit_strategy": cand.ego_input_edit_strategy,
        },
        "retrieval_suggestion": None,
        "scene_id": data.scene_ref.scene_id,
        "scene_token": data.scene_ref.scene_token,
        "candidate_id": data.candidate_id,
        "target_scene": data.scenario_result.target_scene,
        "failure_context": fc.model_dump(mode="json"),
        "retrieved_bev_image": "mock://tool04/retrieved_bev.png",
        "attack_input": attack_input,
        "evidence": {"mode": "mock"},
    }).model_dump(mode="json")
    return {"status": "success", "tool": "edit_planner", "result": result, "meta": {"mode": "mock"}}
