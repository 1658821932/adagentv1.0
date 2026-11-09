"""Interface tests only: scripted backends are not trajectory algorithms."""
import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from tools.registry import call_tool
from mcp_servers.scene_editing.adapter.schemas import AttackInput, EgoInput

@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAGENT_RUNTIME_ROOT",str(tmp_path/"runtime"))
    root=tmp_path/"scene";(root/"metadata").mkdir(parents=True)
    frames=[{"timestamp":1_000_000+i*500_000,"token":str(i),"ego2global_translation":[i,0,0],"anns":{"track_tokens":["vehicle_1"]}} for i in range(3)]
    for name,value in [("frames",frames),("cameras",{"CAM_F0":{}})]:
        (root/"metadata"/(name+".json")).write_text(json.dumps(value))
    (root/"binding.json").write_text("{}")
    return {"scene_id":"log-anchor","scene_token":"anchor","sample_root":str(root),"dataset":"test",
        "matched_camera":"CAM_F0","frame_count":3,"fps":2.0,"manifest_path":str(root/"binding.json"),
        "frames_path":str(root/"metadata/frames.json"),"cameras_path":str(root/"metadata/cameras.json"),
        "cameras":{"CAM_F0":{"video_path":None,"frame_count":3,"available_images":0,"images_complete":False}},"images_complete":False}


def attack_args(source):
    return {"scene_ref":source,"instruction":"Make the annotated lead vehicle slow down.","operation":"edit_existing","target_track":"vehicle_1"}


def test_mock_handoff_from_retrieval_reference_to_attack_and_ego(source):
    a=call_tool("attack_vehicle_editor",attack_args(source),mode="mock")
    assert a["status"]=="success"
    e=call_tool("ego_trajectory_editor",{"attack_result":a,"target_effect":"Brake and stop safely behind the lead vehicle."},mode="mock")
    assert e["status"]=="success"
    assert e["result"]["frozen_scene_path"]==a["result"]["frozen_scene_path"]
    assert e["result"]["target_track"]=="vehicle_1"
    assert e["result"]["ilft_training_data"] is None
    assert e["result"]["simulation_input"] is None
    assert e["result"]["checks"]["physical_avoidance"]=="undetermined"
    assert not Path(e["result"]["expert_trajectory_path"]).exists()


def test_existing_target_can_be_selected_inside_backend(source):
    args=attack_args(source);args.pop("target_track")
    assert AttackInput.model_validate(args).target_track is None

@pytest.mark.parametrize("changes",[
    {"operation":"remove_vehicle"},
    {"operation":"add_vehicle","target_track":None},
    {"maneuver":"left_cut_in"},
    {"unknown_parameter":1},
    {"constraints":{"target_window_s":[8.0,3.0]}},
])
def test_invalid_editing_inputs_are_rejected(source,changes):
    result=call_tool("attack_vehicle_editor",{**attack_args(source),**changes},mode="real")
    assert result["error"]["type"]=="input_validation_error"

@pytest.mark.parametrize("maneuver",["stationary","left_cut_in"])
def test_added_vehicle_requires_explicit_asset_dimensions_and_maneuver(source,maneuver):
    args={"scene_ref":source,"instruction":"Add a vehicle ahead.","operation":"add_vehicle","maneuver":maneuver,"vehicle":{"asset_path":"/assets/car.ply","size_m":[4.4,2.0,1.6]}}
    result=call_tool("attack_vehicle_editor",args,mode="mock")
    assert result["status"]=="success"
    assert result["result"]["target_track"]=="synthetic_vehicle_01"


def test_real_missing_backend_is_explicit_and_task_manifest_is_error(source,tmp_path):
    with patch("mcp_servers.scene_editing.adapter.service.stage_command",return_value=None):
        result=call_tool("attack_vehicle_editor",attack_args(source),mode="real")
    assert result["error"]["type"]=="editing_backend_not_configured"
    manifests=list((tmp_path/"runtime/scene_editing").glob("*/manifest.json"))
    assert len(manifests)==1
    assert json.loads(manifests[0].read_text())["status"]=="error"


def test_failed_or_wrong_upstream_envelope_is_not_accepted(source):
    a=call_tool("attack_vehicle_editor",attack_args(source),mode="mock")
    a["tool"]="another_tool"
    with pytest.raises(ValueError,match="successful attack"):
        EgoInput.model_validate({"attack_result":a,"target_effect":"stop"})

@pytest.fixture
def scripted_backend(tmp_path):
    # Exercise the actual JSON subprocess boundary using a labelled test stub.
    p=tmp_path/"scripted_backend.py"
    p.write_text(r'''import json,sys,shutil,csv
from pathlib import Path
p=json.load(sys.stdin);a=p["arguments"];out=Path(p["output_dir"])
def artifact(name,value):
 f=out/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(value));return str(f)
checks={"reference_screen":"passed","checks":{"scripted_contract":"passed","dynamics":"not_checked"},"report_path":artifact("checks.json",{"test_stub":True})}
if p["stage"]=="attack_editing":
 s=a["scene_ref"];frozen=out/"frozen_scene";(frozen/"metadata").mkdir(parents=True)
 shutil.copy2(s["frames_path"],frozen/"metadata/frames.json")
 traj=out/"attack.csv";traj.write_text("time_s,target_x_m,target_y_m,target_yaw_rad\n0,1,0,0\n")
 result={"status":"completed","mode":"real","run_dir":"backend-owned","source_scene_ref":s,"scene_id":s["scene_id"],"scene_token":s["scene_token"],"target_track":a.get("target_track") or "vehicle_1","selected_candidate_id":"candidate_001","frozen_scene_path":str(frozen),"frames_path":str(frozen/"metadata/frames.json"),"trajectory_path":str(traj),"provenance_path":artifact("freeze-info.json",{"test_stub":True}),"submission_path":artifact("accepted-submission.json",{"candidate_id":"candidate_001"}),"checks":checks,"annotation_changes":{"operation":"update_actor"}}
else:
 attack=a["attack_result"];frames=json.loads(Path(attack["frames_path"]).read_text())
 traj=out/"expert.csv"
 with traj.open("w") as f:
  w=csv.writer(f);w.writerow(["time_s","ego_x_m","ego_y_m","ego_yaw_rad"])
  for frame in frames:w.writerow([(frame["timestamp"]-frames[0]["timestamp"])/1000000,0,0,0])
 result={"status":"completed","mode":"real","run_dir":"backend-owned","scene_id":attack["scene_id"],"scene_token":attack["scene_token"],"attack_run_dir":attack["run_dir"],"frozen_scene_path":attack["frozen_scene_path"],"target_track":attack["target_track"],"selected_candidate_id":"foci_001","selected_source":"foci","expert_trajectory_path":str(traj),"trajectory_metadata_path":artifact("trajectory-meta.json",{"coordinate_frame":"initial_ego_rear_axle","position_reference":"ego_rear_axle","length_unit":"m","angle_unit":"rad","time_unit":"s"}),"conflict_brief_path":artifact("conflict-brief.json",{"directions":[]}),"submission_path":artifact("accepted-submission.json",{"candidate_id":"foci_001"}),"checks":checks,"annotation_changes":{"operation":"export_ego_trajectory","frames_json_modified":False}}
print(json.dumps(result))
''')
    return [sys.executable,str(p)]


def test_configured_json_bridge_keeps_identity_and_outputs_real_files(source,scripted_backend):
    before=Path(source["frames_path"]).read_bytes()
    with patch("mcp_servers.scene_editing.adapter.service.stage_command",return_value=scripted_backend):
        attack=call_tool("attack_vehicle_editor",attack_args(source),mode="real")
        assert attack["status"]=="success",attack
        ego=call_tool("ego_trajectory_editor",{"attack_result":attack,"target_effect":"Brake safely."},mode="real")
    assert ego["status"]=="success",ego
    assert Path(ego["result"]["expert_trajectory_path"]).is_file()
    assert ego["result"]["attack_run_dir"]==attack["result"]["run_dir"]
    assert Path(source["frames_path"]).read_bytes()==before
    assert Path(attack["result"]["run_dir"])/"output" in Path(attack["result"]["frames_path"]).parents
    assert ego["result"]["ilft_training_data"] is None
    assert ego["result"]["simulation_input"] is None


def test_mock_expert_csv_cannot_be_passed_as_il_labels(source):
    attack=call_tool("attack_vehicle_editor",attack_args(source),mode="mock")
    ego=call_tool("ego_trajectory_editor",{"attack_result":attack,"target_effect":"Stop."},mode="mock")
    from tools.tool10_policy_trainer.schemas import Input
    with pytest.raises(ValueError):
        Input.model_validate({"training_mode":"ilft","training_data":{"expert_trajectory_path":ego["result"]["expert_trajectory_path"]},"base_checkpoint":"/base.pth"})


def test_optional_handoff_blocks_match_existing_downstream_schemas():
    from mcp_servers.scene_editing.adapter.schemas import SimulationInput,ILTrainingData
    from tools.tool08_policy_runner.schemas import Input as SimInput
    from tools.tool10_policy_trainer.schemas import Input as TrainInput
    s=SimulationInput(scenario_data_folder_path="/native/scenes",asset_folder_path="/native/assets")
    SimInput.model_validate({**s.model_dump(),"config_path":"/cfg.py","checkpoint_path":"/base.pth","model_name":"base","data_type":"custom"})
    labels=ILTrainingData(ann_file="/labels/infos.pkl",token_file="/labels/tokens.yaml")
    TrainInput.model_validate({"training_mode":"ilft","training_data":labels.model_dump(),"base_checkpoint":"/base.pth"})


@pytest.mark.parametrize("fault,expected",[
    ("wrong_scene","editing_identity_mismatch"),
    ("no_submission","editing_not_completed"),
    ("missing_file","editing_artifact_missing"),
    ("bad_timeline","editing_identity_mismatch"),
    ("source_mutation","editing_input_modified"),
])
def test_attack_bridge_rejects_invalid_backend_artifacts(source,scripted_backend,fault,expected):
    import subprocess
    from mcp_servers.scene_editing.adapter import service
    original = service.run_logged
    def altered(*args,**kwargs):
        completed=original(*args,**kwargs)
        result=json.loads(completed.stdout)
        if fault=="wrong_scene":result["scene_token"]="another-scene"
        elif fault=="no_submission":result["status"]="no_submission"
        elif fault=="missing_file":Path(result["trajectory_path"]).unlink()
        elif fault=="bad_timeline":
            p=Path(result["frames_path"]);frames=json.loads(p.read_text());frames[-1]["timestamp"]+=100;p.write_text(json.dumps(frames))
        elif fault=="source_mutation":Path(source["frames_path"]).write_text("[]")
        return subprocess.CompletedProcess(completed.args,completed.returncode,json.dumps(result),completed.stderr)
    with patch.object(service,"stage_command",return_value=scripted_backend),patch.object(service,"run_logged",side_effect=altered):
        result=call_tool("attack_vehicle_editor",attack_args(source),mode="real")
    assert result["error"]["type"]==expected,result

@pytest.mark.parametrize("fault,expected",[
    ("wrong_target","editing_identity_mismatch"),
    ("bad_time","editing_identity_mismatch"),
    ("bad_coordinates","editing_output_invalid"),
    ("not_finite","editing_output_invalid"),
])
def test_ego_bridge_rejects_wrong_attack_or_trajectory(source,scripted_backend,fault,expected):
    import subprocess
    from mcp_servers.scene_editing.adapter import service
    with patch.object(service,"stage_command",return_value=scripted_backend):
        attack=call_tool("attack_vehicle_editor",attack_args(source),mode="real")
    assert attack["status"]=="success",attack
    original=service.run_logged
    def altered(*args,**kwargs):
        completed=original(*args,**kwargs);result=json.loads(completed.stdout)
        if fault=="wrong_target":result["target_track"]="another-track"
        elif fault=="bad_coordinates":Path(result["trajectory_metadata_path"]).write_text(json.dumps({"coordinate_frame":"world"}))
        elif fault=="bad_time":
            p=Path(result["expert_trajectory_path"]);p.write_text(p.read_text().replace("0.5,0,0,0","0.6,0,0,0"))
        elif fault=="not_finite":
            p=Path(result["expert_trajectory_path"]);p.write_text(p.read_text().replace("0.5,0,0,0","0.5,nan,0,0"))
        return subprocess.CompletedProcess(completed.args,completed.returncode,json.dumps(result),completed.stderr)
    with patch.object(service,"stage_command",return_value=scripted_backend),patch.object(service,"run_logged",side_effect=altered):
        result=call_tool("ego_trajectory_editor",{"attack_result":attack,"target_effect":"Stop safely."},mode="real")
    assert result["error"]["type"]==expected,result


@pytest.mark.parametrize("checks",[{}, {"motion":"failed"}, {"motion":"not_checked"}])
def test_accepted_candidate_cannot_hide_failed_or_missing_checks(checks):
    from mcp_servers.scene_editing.adapter.schemas import StageChecks
    with pytest.raises(ValueError):
        StageChecks.model_validate({"reference_screen":"passed","checks":checks,"report_path":"/report.json"})



def test_saved_result_references_resolve_before_json_backend(source,scripted_backend,tmp_path):
    retrieved=tmp_path/"retrieved.json"
    retrieved.write_text(json.dumps({"tool":"video_retriever","status":"success","result":{"scene_ref":source}}))
    args=attack_args(source);args["scene_ref"]=str(retrieved)
    with patch("mcp_servers.scene_editing.adapter.service.stage_command",return_value=scripted_backend):
        attack=call_tool("attack_vehicle_editor",args,mode="real")
        assert attack["status"]=="success",attack
        saved=tmp_path/"attack.json";saved.write_text(json.dumps(attack))
        ego=call_tool("ego_trajectory_editor",{"attack_result":str(saved),"target_effect":"Stop safely."},mode="real")
    assert ego["status"]=="success",ego
    assert ego["result"]["attack_run_dir"]==attack["result"]["run_dir"]


@pytest.mark.parametrize("payload",[
    {"tool":"video_retriever","status":"error","result":{}},
    {"tool":"wrong","status":"success","result":{"scene_ref":{}}},
    {"tool":"video_retriever","status":"success","result":{}},
    {"tool":"video_retriever","status":"success","result":[]},
])
def test_invalid_reference_files_become_input_errors(source,tmp_path,payload):
    p=tmp_path/"bad.json";p.write_text(json.dumps(payload))
    args=attack_args(source);args["scene_ref"]=str(p)
    result=call_tool("attack_vehicle_editor",args,mode="real")
    assert result["error"]["type"]=="input_validation_error",result
