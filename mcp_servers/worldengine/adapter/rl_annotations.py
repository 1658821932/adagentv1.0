"""Package actual WE rollout annotations for RLFT; never invent actions or rewards."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import pickle
import sys
import numpy as np

if not __package__:
    # The adapter launches this file directly in the AlgEngine environment.
    # Its shared score reader lives under the repository root, not this folder.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from .pdm_filter import load_scene_scores, failed_scene_anchors
except ImportError:
    from pdm_filter import load_scene_scores, failed_scene_anchors

REWARDS=("score","no_at_fault_collisions","drivable_area_compliance",
         "ego_progress","time_to_collision_within_bound","comfort")
VERSION="rlft_annotations.v1"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def file(path):
    p=Path(path).expanduser().resolve()
    if not p.is_file():raise FileNotFoundError(str(p))
    return p

def package(rollout_dir, vocab_path, output_dir, *, config_path=None, checkpoint_path=None):
    rollout=Path(rollout_dir).expanduser().resolve()
    if checkpoint_path:checkpoint_path=file(checkpoint_path)
    vocab_path=file(vocab_path)
    vocab=np.load(vocab_path,allow_pickle=False,mmap_mode="r")
    if vocab.shape!=(8192,40,3) or not np.isfinite(vocab).all():
        raise ValueError("RL vocabulary must be finite (8192,40,3)")
    vocab_sha=digest(vocab_path)
    if config_path:
        from mmcv import Config
        config=Config.fromfile(str(file(config_path)),import_custom_modules=False)
        model_vocab=file(config.model.planning_head.vocab_path)
        if digest(model_vocab)!=vocab_sha:
            raise ValueError("Policy vocabulary differs from the native candidate reward vocabulary")
    score_map,scores_file=load_scene_scores(rollout)
    if any(not np.isfinite(s) or not 0<=s<=1 for s in score_map.values()):
        raise ValueError("Official scene PDM scores must be finite and within [0,1]")
    sources=[rollout/"meta_datas/combined.pkl"] if (rollout/"meta_datas/combined.pkl").is_file() else sorted((rollout/"meta_datas").glob("*.pkl"))
    if not sources:raise ValueError("No native rollout frame annotations")
    infos=[]
    for path in sources:
        with file(path).open("rb") as f:rows=pickle.load(f)
        if not isinstance(rows,list):raise ValueError("Rollout metadata must contain a frame list")
        infos.extend(rows)
    plan_path=file(rollout/"pdms_pkl/plan_idx.csv")
    plans={}
    with plan_path.open(newline="") as f:
        for row in csv.DictReader(f):
            key=(row["prefix"],int(row["step"])-1)  # Official loader uses step-1 == frame_idx.
            idx=int(row["plan_idx"])
            if key in plans or not 0<=idx<8192:
                raise ValueError("Duplicate or out-of-range actual plan index")
            plans[key]=idx
    by_scene={}
    seen=set()
    for row in infos:
        token=row["token"]
        if token in seen:raise ValueError("Duplicate rollout frame token")
        seen.add(token)
        by_scene.setdefault(row["log_name"],[]).append(row)
    anchors={name.split("-",2)[-1] for name in by_scene}
    scored_anchors={name.split("-",2)[-1] for name in score_map}
    if not anchors<=scored_anchors:
        raise ValueError("Rollout scenes are missing matching official PDM scores")
    eligible=failed_scene_anchors(score_map,anchors,max_score=0.5)
    index=[]
    reward_count=0
    solvability={}
    for name,rows in by_scene.items():
        rows.sort(key=lambda r:int(r["frame_idx"]))
        if [int(r["frame_idx"]) for r in rows]!=list(range(len(rows))) or len(rows)<12:
            raise ValueError("RL rollout needs contiguous history/future frames (at least 12)")
        times=np.asarray([r["timestamp"] for r in rows],float)
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
            raise ValueError("Frame timestamps must be finite and increasing")
        cameras=set(rows[0]["cams"])
        if not cameras:raise ValueError("No camera annotations")
        anchor=name.split("-",2)[-1]
        solvability.setdefault(anchor, {"solvable_frames": 0, "max_score": 0.0, "max_safe_vocab_count": 0, "evaluated_frames": 0})
        for row in rows:
            step=int(row["frame_idx"])
            if set(row["cams"])!=cameras:raise ValueError("Camera set changes within a scene")
            pose=np.asarray(row["ego2global"])
            if pose.shape!=(4,4) or not np.isfinite(pose).all():
                raise ValueError("Invalid actual ego pose")
            for key,shape in (
                ("gt_pre_bbox_sdc_lidar",(1,3,9)),("gt_pre_bbox_sdc_global",(1,3,9)),
                ("gt_fut_bbox_sdc_lidar",(1,8,9)),("gt_fut_bbox_sdc_global",(1,8,9)),
                ("gt_pre_bbox_sdc_mask",(1,3,1)),("gt_fut_bbox_sdc_mask",(1,8,1)),
                ("can_bus",(18,)),("driving_command",(4,)),("lidar2ego",(4,4))):
                array=np.asarray(row[key])
                if array.shape!=shape or not np.isfinite(array).all():
                    raise ValueError("Missing/invalid native training label: "+key)
                row[key]=array
            images={}
            for camera,data in row["cams"].items():
                for key,shape in (("sensor2lidar_rotation",(3,3)),("sensor2lidar_translation",(3,)),("cam_intrinsic",(3,3))):
                    array=np.asarray(data[key])
                    if array.shape!=shape or not np.isfinite(array).all():
                        raise ValueError("Invalid training camera calibration: "+key)
                    data[key]=array
                distortion=np.asarray(data["distortion"])
                if distortion.ndim!=1 or not np.isfinite(distortion).all():
                    raise ValueError("Invalid camera distortion")
                path=file(rollout/"sensor_blobs"/data["data_path"])
                if not path.is_relative_to((rollout/"sensor_blobs").resolve()):
                    raise ValueError("Camera image escapes rollout directory")
                images[camera]=str(path)
            anns=row["anns"]
            tracks=list(map(str,anns["track_tokens"]))
            if len(tracks)!=len(set(tracks)):
                raise ValueError("Duplicate actor identity in frame")
            boxes=np.asarray(anns["gt_boxes"])
            velocities=np.asarray(anns["gt_velocity_3d"])
            if not tracks and boxes.size==0 and velocities.size==0:
                boxes=boxes.reshape(0,7);velocities=velocities.reshape(0,3)
            if len(anns["gt_names"])!=len(tracks) or boxes.shape!=(len(tracks),7) or velocities.shape!=(len(tracks),3):
                raise ValueError("Actor boxes, velocities and identities are misaligned")
            if not np.isfinite(boxes).all() or not np.isfinite(velocities).all() or np.any(boxes[:,3:6]<=0):
                raise ValueError("Invalid edited actor geometry/velocity")
            sample=dict(token=row["token"],scene=name,frame_idx=step,timestamp_us=int(row["timestamp"]),
                        cameras=images,actor_count=len(tracks),invalid=bool(row.get("invalid",False)))
            if step>=3 and not row.get("invalid",False):
                key=(anchor,step)
                if key not in plans:raise ValueError("Missing actual selected plan for reward frame")
                reward_path=file(rollout/"pdms_pkl"/(name+"_step_"+str(step)+"_scores.pkl"))
                with reward_path.open("rb") as f:rewards=pickle.load(f)
                for field in REWARDS:
                    values=np.asarray(rewards[field])
                    if values.shape!=(8192,) or not np.isfinite(values).all():
                        raise ValueError("Invalid candidate reward vector: "+field)
                sc_vec=np.asarray(rewards["score"],dtype=float)
                nc_vec=np.asarray(rewards["no_at_fault_collisions"],dtype=float)
                dac_vec=np.asarray(rewards["drivable_area_compliance"],dtype=float)
                f_max=float(sc_vec.max())
                f_safe=int(((nc_vec>=0.99)&(dac_vec>=0.99)).sum())
                sv=solvability[anchor]
                sv["evaluated_frames"]+=1
                sv["max_score"]=max(sv["max_score"],round(f_max,4))
                sv["max_safe_vocab_count"]=max(sv["max_safe_vocab_count"],f_safe)
                if f_max>=0.3 and f_safe>=5:
                    sv["solvable_frames"]+=1
                sample.update(reward_file=str(reward_path),reward_sha256=digest(reward_path),
                              actual_plan_idx=plans[key],plan_csv_step=step+1,
                              vocab_max_score=round(f_max,4),vocab_safe_count=f_safe)
                reward_count+=1
            index.append(sample)
    unsolvable_anchors=set()
    for anc, sv in solvability.items():
        sv["solvable"] = bool(sv["solvable_frames"] >= 2 or (sv["evaluated_frames"] == 1 and sv["solvable_frames"] >= 1))
        if anc in eligible and sv["evaluated_frames"] > 0 and not sv["solvable"]:
            unsolvable_anchors.add(anc)
    eligible = eligible - unsolvable_anchors
    infos=[row for name in sorted(by_scene) for row in by_scene[name]]
    # Validate everything before writing a new package. Source rollout stays unchanged.
    out=Path(output_dir).resolve()
    out.mkdir(parents=True,exist_ok=False)
    annotations=out/"annotations.pkl"
    with annotations.open("xb") as f:pickle.dump(infos,f,protocol=4)
    tokens=out/"tokens.json"
    tokens.write_text(json.dumps({"tokens":sorted(eligible)},indent=2))
    all_tokens=out/"all_scene_tokens.json"
    all_tokens.write_text(json.dumps({"tokens":sorted(anchors)},indent=2))
    manifest_path=out/"manifest.json"
    manifest=dict(schema_version=VERSION,status="ready" if eligible else "no_eligible_scenes",
        rollout_dir=str(rollout),annotation_file=str(annotations),
        annotation_sha256=digest(annotations),metadata_sources=[dict(path=str(p.resolve()),sha256=digest(p)) for p in sources],
        token_file=str(tokens),token_sha256=digest(tokens),all_scene_tokens=str(all_tokens),
        vocab_path=str(vocab_path),vocabulary_sha256=vocab_sha,
        plan_idx_file=str(plan_path),plan_idx_sha256=digest(plan_path),
        pdm_scores_file=str(scores_file.resolve()),pdm_scores_sha256=digest(scores_file),
        frame_count=len(infos),scene_count=len(by_scene),reward_frame_count=reward_count,
        eligible_scene_count=len(eligible),eligible_scene_anchors=sorted(eligible),
        selection_rule="official closed-loop PDM total score < 0.5 and 8192-vocab solvable",
        label_source="actual WE simulation annotations and candidate rewards",
        frame_to_plan_alignment="plan_idx.csv step - 1 == metadata frame_idx",
        config_path=str(file(config_path)) if config_path else None,
        checkpoint_path=str(file(checkpoint_path)) if checkpoint_path else None,
        samples=index)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    training_data=dict(rollout_dir=str(rollout),token_file=str(tokens),vocab_path=str(vocab_path),manifest_path=str(manifest_path)) if eligible else None
    return dict(rlft_training_data=training_data,rlft_data_report=dict(
        status=manifest["status"],manifest_path=str(manifest_path),frame_count=len(infos),
        reward_frame_count=reward_count,scene_count=len(by_scene),eligible_scene_count=len(eligible),
        vocab_solvability=solvability,
        unsolvable_dead_end_scenes=sorted(unsolvable_anchors),
        reason=None if eligible else (
            "PDM-zero scene rejected: 8192-trajectory vocabulary has no solvable high-reward avoidance path (dead-end attack)."
            if unsolvable_anchors else
            "No official low-score (PDM < 0.5) scene; annotations were exported, training selection is empty."
        )))

def verify_manifest(data):
    """Validate provenance before loading the exported annotations for training."""
    manifest=json.loads(file(data["manifest_path"]).read_text())
    if manifest.get("schema_version")!=VERSION or manifest.get("status")!="ready":
        raise ValueError("RL annotation package has no eligible training scenes")
    if Path(manifest["rollout_dir"]).resolve()!=Path(data["rollout_dir"]).resolve():
        raise ValueError("RL manifest belongs to another rollout")
    for argument,key in (("vocab_path","vocabulary_sha256"),("token_file","token_sha256")):
        if digest(file(data[argument]))!=manifest[key]:
            raise ValueError("RL package input changed: "+argument)
    for path_key,hash_key in (("annotation_file","annotation_sha256"),("plan_idx_file","plan_idx_sha256"),("pdm_scores_file","pdm_scores_sha256")):
        if digest(file(manifest[path_key]))!=manifest[hash_key]:
            raise ValueError("RL annotation package source changed: "+path_key)
    for source in manifest["metadata_sources"]:
        if digest(file(source["path"]))!=source["sha256"]:
            raise ValueError("RL rollout metadata changed after packaging")
    for sample in manifest["samples"]:
        for path in sample["cameras"].values():file(path)
        if "reward_file" in sample and digest(file(sample["reward_file"]))!=sample["reward_sha256"]:
            raise ValueError("Candidate rewards changed after packaging")
    return manifest

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--request",required=True)
    args=parser.parse_args()
    result=package(**json.loads(Path(args.request).read_text()))
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":main()
