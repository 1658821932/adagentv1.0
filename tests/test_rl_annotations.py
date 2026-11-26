"""RL annotation/plan/reward linkage without model or GPU calls."""
from pathlib import Path
import csv,json,pickle
import pytest
np=pytest.importorskip("numpy")
from mcp_servers.worldengine.adapter.rl_annotations import package,verify_manifest,REWARDS


def test_direct_export_entrypoint_without_repository_pythonpath(tmp_path):
    import os
    import subprocess
    import sys
    import mcp_servers.worldengine.adapter.rl_annotations as module

    script = Path(module.__file__).resolve()
    repository = script.parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in sys.path if path and Path(path).resolve() != repository
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--request" in completed.stdout


@pytest.fixture
def rollout(tmp_path):
    root=tmp_path/"rollout"
    for folder in ("meta_datas","pdms_pkl","sensor_blobs"):(root/folder).mkdir(parents=True)
    rows=[]
    for i in range(12):
        cams={}
        for camera in ("CAM_F0","CAM_L0","CAM_R0","CAM_L1","CAM_R1","CAM_L2","CAM_R2","CAM_B0"):
            path=Path(camera)/(str(i)+".jpg");target=root/"sensor_blobs"/path
            target.parent.mkdir(exist_ok=True);target.write_bytes(b"unit-test-image")
            cams[camera]={"data_path":str(path),"sensor2lidar_rotation":np.eye(3),"sensor2lidar_translation":np.zeros(3),"cam_intrinsic":np.eye(3),"distortion":[0.,0.,0.,0.,0.]}
        rows.append(dict(token="frame-"+str(i),frame_idx=i,timestamp=1000000+i*500000,
            log_name="log-clip-anchor",ego2global=np.eye(4),cams=cams,
            anns=dict(gt_boxes=np.array([[1.,2.,0.,4.,2.,1.5,0.]]),
                      gt_velocity_3d=np.zeros((1,3)),gt_names=["vehicle"],track_tokens=["edited-car"])))
    for row in rows:
        row.update(lidar2ego=np.eye(4),can_bus=np.zeros(18),driving_command=np.array([1.,0.,0.,0.]))
        for prefix,length in (("pre",3),("fut",8)):
            for frame in ("lidar","global"):
                row["gt_"+prefix+"_bbox_sdc_"+frame]=np.zeros((1,length,9))
            row["gt_"+prefix+"_bbox_sdc_mask"]=np.ones((1,length,1))
    with (root/"meta_datas/scene.pkl").open("wb") as f:pickle.dump(rows,f)
    (root/"all_scenes_pdm_averages_NR.csv").write_text("token,score\nlog-clip-anchor,0.0\n")
    with (root/"pdms_pkl/plan_idx.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["prefix","step","plan_idx"]);writer.writeheader()
        for step in range(3,12):writer.writerow(dict(prefix="anchor",step=step+1,plan_idx=7))
    for step in range(3,12):
        with (root/"pdms_pkl"/("log-clip-anchor_step_"+str(step)+"_scores.pkl")).open("wb") as f:
            pickle.dump({key:np.ones(8192) for key in REWARDS},f)
    vocab=tmp_path/"vocab.npy";np.save(vocab,np.zeros((8192,40,3),dtype=np.float32))
    return root,vocab,tmp_path/"package"

def test_ready_package_and_actual_plan_offset(rollout):
    root,vocab,out=rollout;source=(root/"meta_datas/scene.pkl").read_bytes()
    result=package(root,vocab,out);data=result["rlft_training_data"]
    manifest=verify_manifest(data)
    assert manifest["frame_count"]==12 and manifest["reward_frame_count"]==9
    assert manifest["eligible_scene_anchors"]==["anchor"]
    with Path(manifest["annotation_file"]).open("rb") as f:rows=pickle.load(f)
    assert rows[3]["anns"]["track_tokens"]==["edited-car"]
    np.testing.assert_array_equal(rows[3]["anns"]["gt_boxes"],[[1,2,0,4,2,1.5,0]])
    sample=manifest["samples"][3]
    assert sample["frame_idx"]==3 and sample["plan_csv_step"]==4 and sample["actual_plan_idx"]==7
    assert len(sample["cameras"])==8
    assert (root/"meta_datas/scene.pkl").read_bytes()==source
    from common.rl_training import RLTrainingData
    from tools.tool08_policy_trainer.schemas import Input
    RLTrainingData.model_validate(data)
    assert Input.model_validate(dict(training_mode="rlft",training_data=data,base_checkpoint="/base.pth")).training_data.manifest_path==str(out/"manifest.json")

def test_nonzero_score_exports_annotations_without_faking_readiness(rollout):
    root,vocab,out=rollout;scores=root/"all_scenes_pdm_averages_NR.csv"
    scores.write_text("token,score\nlog-clip-anchor,0.79183\n")
    result=package(root,vocab,out)
    assert result["rlft_training_data"] is None
    assert result["rlft_data_report"]["status"]=="no_eligible_scenes"
    assert (out/"annotations.pkl").is_file()
    assert json.loads((out/"tokens.json").read_text())=={"tokens":[]}
    assert "0.79183" in scores.read_text()

@pytest.mark.parametrize("fault",["reward","plan","actors","image"])
def test_misaligned_inputs_rejected(rollout,fault):
    root,vocab,out=rollout
    if fault=="reward":
        path=root/"pdms_pkl/log-clip-anchor_step_3_scores.pkl"
        with path.open("rb") as f:d=pickle.load(f)
        d["score"]=np.ones(8)
        with path.open("wb") as f:pickle.dump(d,f)
    elif fault=="plan":
        path=root/"pdms_pkl/plan_idx.csv";lines=path.read_text().splitlines()
        path.write_text("\n".join([lines[0]]+lines[2:])+"\n")
    elif fault=="actors":
        path=root/"meta_datas/scene.pkl"
        with path.open("rb") as f:rows=pickle.load(f)
        rows[0]["anns"]["gt_velocity_3d"]=np.zeros((2,3))
        with path.open("wb") as f:pickle.dump(rows,f)
    else:(root/"sensor_blobs/CAM_F0/0.jpg").unlink()
    with pytest.raises((ValueError,FileNotFoundError)):package(root,vocab,out)
    assert not out.exists()

def test_changed_rewards_rejected_at_training_handoff(rollout):
    root,vocab,out=rollout;data=package(root,vocab,out)["rlft_training_data"]
    path=root/"pdms_pkl/log-clip-anchor_step_3_scores.pkl"
    with path.open("rb") as f:d=pickle.load(f)
    d["score"][0]=.123
    with path.open("wb") as f:pickle.dump(d,f)
    with pytest.raises(ValueError,match="rewards changed"):verify_manifest(data)

def test_wrong_rollout_rejected(rollout):
    root,vocab,out=rollout;data=package(root,vocab,out)["rlft_training_data"]
    with pytest.raises(ValueError,match="another rollout"):
        verify_manifest({**data,"rollout_dir":str(root.parent/"other")})

def test_empty_actor_annotations_are_valid(rollout):
    root,vocab,out=rollout;path=root/"meta_datas/scene.pkl"
    with path.open("rb") as f:rows=pickle.load(f)
    for row in rows:row["anns"]=dict(gt_boxes=[],gt_velocity_3d=[],gt_names=[],track_tokens=[])
    with path.open("wb") as f:pickle.dump(rows,f)
    assert package(root,vocab,out)["rlft_data_report"]["status"]=="ready"


def test_missing_native_training_trajectory_is_not_fabricated(rollout):
    root,vocab,out=rollout;path=root/"meta_datas/scene.pkl"
    with path.open("rb") as f:rows=pickle.load(f)
    del rows[3]["gt_fut_bbox_sdc_lidar"]
    with path.open("wb") as f:pickle.dump(rows,f)
    with pytest.raises(KeyError):package(root,vocab,out)
    assert not out.exists()
