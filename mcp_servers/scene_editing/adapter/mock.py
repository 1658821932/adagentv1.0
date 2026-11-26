"""Deterministic interface examples; paths are placeholders, no real artifacts."""
from pathlib import Path
from .schemas import AttackInput, AttackOutput, EgoInput, EgoOutput

def attack(arguments):
    data=AttackInput.model_validate(arguments)
    operation=data.operation;track=data.target_track;choice=None
    if operation=='auto':
        can_add = data.vehicle is not None or data.vehicle_catalog_class is not None
        option=next(option for option in data.plan_options if option.operation=='edit_existing' or can_add)
        operation=option.operation;track=option.target_track
        choice=dict(option_id=option.option_id,operation=operation,reason='Deterministic mock selection; not an evaluated scene decision')
    root=Path("/mock/scene_editing/attack")
    result={"mode":"mock","run_dir":str(root),"operation":operation,"plan_choice":choice,"source_scene_ref":data.scene_ref.model_dump(),
        "scene_id":data.scene_ref.scene_id,"scene_token":data.scene_ref.scene_token,
        "target_track":track or ("mock_existing_track" if operation=="edit_existing" else "synthetic_vehicle_01"),"selected_candidate_id":"candidate_001",
        "frozen_scene_path":str(root/"frozen_scene"),"frames_path":str(root/"frozen_scene/metadata/frames.json"),
        "trajectory_path":str(root/"trajectory.csv"),"provenance_path":str(root/"freeze-info.json"),
        "submission_path":str(root/"accepted-submission.json"),
        "checks":{"reference_screen":"passed","checks":{"mock_contract":"passed","official_road":"not_checked"},"report_path":str(root/"checks.json")},
        "annotation_changes":{"operation":"append_actor" if operation=="add_vehicle" else "update_actor","fields":["anns.gt_boxes","anns.gt_velocity_3d"]}}
    return AttackOutput.model_validate(result).model_dump()

def ego(arguments):
    data=EgoInput.model_validate(arguments);a=data.attack_result;root=Path("/mock/scene_editing/ego")
    return EgoOutput.model_validate({"mode":"mock","run_dir":str(root),"scene_id":a.scene_id,"scene_token":a.scene_token,
        "attack_run_dir":a.run_dir,"frozen_scene_path":a.frozen_scene_path,"target_track":a.target_track,
        "selected_candidate_id":"constructor_001","selected_source":"constructor",
        "expert_trajectory_path":str(root/"trajectory.csv"),"trajectory_metadata_path":str(root/"trajectory-meta.json"),
        "conflict_brief_path":str(root/"conflict-brief.json"),"submission_path":str(root/"accepted-submission.json"),
        "checks":{"reference_screen":"passed","checks":{"mock_contract":"passed","physical_tracking":"not_checked"},"report_path":str(root/"checks.json")},
        "annotation_changes":{"operation":"export_ego_trajectory","fields":["ego_x_m","ego_y_m","ego_yaw_rad"],"frames_json_modified":False}}).model_dump()
