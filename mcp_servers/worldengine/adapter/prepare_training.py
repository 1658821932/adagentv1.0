"""Prepare locally generated WE rollouts in an isolated AlgEngine data root."""
from pathlib import Path
import os
import sys
import json
import pickle
import shutil
import hashlib
import csv
import numpy as np
import yaml
from mmcv import Config

REWARDS = ("score", "no_at_fault_collisions", "drivable_area_compliance",
           "ego_progress", "time_to_collision_within_bound", "comfort")

def require_file(p):
    p = Path(p).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p

def link(source, dest):
    source = Path(source).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(source, target_is_directory=source.is_dir())

def prepare(args):
    source_root = Path(args["worldengine_root"])
    work = Path(args["work_dir"])
    data = args["training_data"]
    mode = args["training_mode"]
    runtime = work / "worldengine" if mode == "rlft" else source_root
    os.environ["WORLDENGINE_ROOT"] = str(runtime)
    cfg = Config.fromfile(args["config_path"])
    report = dict(training_mode=mode, runtime_root=str(runtime))
    tokens_path = require_file(data["token_file"])
    tokens = yaml.safe_load(tokens_path.read_text())
    if not isinstance(tokens, dict) or not tokens.get("tokens"):
        raise ValueError("token_file must contain a non-empty tokens list")
    if mode == "rlft":
        if cfg.model.planning_head.type != "TrajScoringHeadRL":
            raise ValueError("rlft requires a TrajScoringHeadRL config")
        if cfg.data.train.type != "NavSimOpenSceneE2EFineTuneSynthetic":
            raise ValueError("rlft requires the synthetic rollout dataset config")
        rollout = Path(data["rollout_dir"]).expanduser().resolve()
        vocab = np.load(require_file(data["vocab_path"]), allow_pickle=False)
        if vocab.shape != (8192, 40, 3) or not np.isfinite(vocab).all():
            raise ValueError("Expected a finite (8192, 40, 3) trajectory vocabulary")
        metadata_root = rollout / "meta_datas"
        combined = metadata_root / "combined.pkl"
        metadata_files = [combined] if combined.is_file() else sorted(metadata_root.glob("*.pkl"))
        if not metadata_files:
            raise ValueError("No rollout metadata files")
        infos = []
        for p in metadata_files:
            # These inputs are locally generated WE artifacts, not arbitrary downloaded pickles.
            with p.open("rb") as f:
                rows = pickle.load(f)
            if not isinstance(rows, list):
                raise ValueError("WE rollout metadata must be a list of frames")
            infos.extend(rows)
        plan_file = require_file(rollout / "pdms_pkl/plan_idx.csv")
        with plan_file.open() as f:
            plans = {(row["prefix"], int(row["step"]) - 1): int(row["plan_idx"])
                     for row in csv.DictReader(f)}
        name = "agent_rollout"
        history = 3
        cache = {}
        reward_frames = selected = 0
        allowed = set(tokens["tokens"])
        valid_by_log = {}
        for info in infos:
            info["syn_name"] = name
            for cam in info["cams"].values():
                image = (rollout / "sensor_blobs" / cam["data_path"]).resolve()
                if not image.is_relative_to((rollout / "sensor_blobs").resolve()):
                    raise ValueError("Camera path escapes rollout sensor_blobs")
                require_file(image)
            log_name = info["log_name"]
            step = int(info["frame_idx"])
            if step < history:
                continue
            reward_file = rollout / "pdms_pkl" / (log_name + "_step_" + str(step) + "_scores.pkl")
            with require_file(reward_file).open("rb") as f:
                rewards = pickle.load(f)
            clean = {}
            for key in REWARDS:
                a = np.asarray(rewards[key], dtype=np.float32)
                if a.shape != (8192,) or not np.isfinite(a).all():
                    raise ValueError("Invalid reward vector: " + key)
                clean[key] = a
            cache[info["token"]] = clean
            reward_frames += 1
            anchor = log_name.split("-", 2)[-1]
            if anchor in allowed and not info.get("invalid", False):
                valid_by_log.setdefault(log_name, []).append(step)
                if data["sample_filter"] == "v2":
                    selected += int(step == history)
                else:
                    plan_idx = plans.get((anchor, step))
                    if plan_idx is None or not 0 <= plan_idx < 8192:
                        raise ValueError("Missing or invalid executed plan index")
                    low_ep = (clean["drivable_area_compliance"][plan_idx] == 1
                        and clean["no_at_fault_collisions"][plan_idx] == 1
                        and clean["ego_progress"][plan_idx] < 0.2)
                    selected += int(clean["score"].max() >= 0.9
                        and (clean["score"][plan_idx] <= 2/3 or low_ep))
        future = 8
        if not valid_by_log or not cache:
            raise ValueError("No matching rollout rewards for requested tokens")
        for log_name in valid_by_log:
            all_steps = sorted(int(i["frame_idx"]) for i in infos if i["log_name"] == log_name)
            if all_steps != list(range(len(all_steps))) or len(all_steps) < history + 1 + future:
                raise ValueError("Each rollout needs contiguous history and future frames (at least 12)")
        if not selected:
            raise ValueError("No training samples remain after " + data["sample_filter"] + " filtering")
        for rel in ["projects", "data/raw/nuplan", "data/alg_engine/ckpts"]:
            p = source_root / rel
            if p.exists():
                link(p, runtime / rel)
        for rel in ["pdms_pkl", "sensor_blobs"]:
            link(rollout / rel, runtime / "data/alg_engine/openscene-synthetic" / rel / name)
        plan = rollout / "pdms_pkl/plan_idx.csv"
        if not plan.is_file():
            raise ValueError("rollout pdms_pkl/plan_idx.csv is required")
        meta = runtime / "data/alg_engine/openscene-synthetic/meta_datas" / name
        meta.mkdir(parents=True)
        with (meta / "combined.pkl").open("wb") as f:
            pickle.dump(infos, f, protocol=4)
        cache_path = runtime / "data/alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain.pkl"
        cache_path.parent.mkdir(parents=True)
        with cache_path.open("wb") as f:
            pickle.dump(cache, f, protocol=4)
        link(Path(data["vocab_path"]).expanduser(), runtime / "data/alg_engine/test_8192_kmeans.npy")
        ann = work / "empty_original_annotations.pkl"
        with ann.open("wb") as f:
            pickle.dump([], f, protocol=4)
        target_tokens = work / "navtrain_tokens.yaml"
        shutil.copy2(tokens_path, target_tokens)
        cfg.data.train.update(ann_file=str(ann), nav_filter_path=str(target_tokens),
            finetune_yaml=[str(target_tokens)], folder_name=[name],
            customized_filter=data["sample_filter"], include_real_failures=False, normal_ratio=0)
        cfg.model.planning_head.update(rl_finetuning=True, reward_shaping=True,
            vocab_path=str(runtime / "data/alg_engine/test_8192_kmeans.npy"))
        report.update(metadata_frames=len(infos), reward_frames=reward_frames,
            selected_samples=selected, sample_filter=data["sample_filter"],
            rollout_dir=str(rollout), reward_source="actual WE candidate scores",
            vocabulary_sha256=hashlib.sha256(Path(data["vocab_path"]).expanduser().read_bytes()).hexdigest())
    else:
        cfg.data.train.update(ann_file=str(require_file(data["ann_file"])),
            nav_filter_path=str(tokens_path), finetune_yaml=[str(tokens_path)])
        cfg.model.planning_head.update(rl_finetuning=False, reward_shaping=False)
        with Path(data["ann_file"]).expanduser().open("rb") as f:
            annotations = pickle.load(f)
        frames = annotations["infos"] if isinstance(annotations, dict) else annotations
        history = int(cfg.data.train.get("queue_length", cfg.get("queue_length", 4))) - 1
        allowed = set(tokens["tokens"])
        eligible = sum(1 for i, row in enumerate(frames)
            if row["token"] in allowed and i >= history
            and frames[i-history].get("log_token", frames[i-history].get("scene_token"))
                == row.get("log_token", row.get("scene_token")))
        if not eligible:
            raise ValueError("IL token selection has no sample with the required history; select a later frame")
        report["eligible_history_samples"] = eligible
    cfg.data.update(samples_per_gpu=1, workers_per_gpu=0)
    cfg.load_from = args["base_checkpoint"]
    cfg.resume_from = None
    cfg.total_epochs = args["epochs"]
    cfg.runner = dict(type="EpochBasedRunner", max_epochs=args["epochs"])
    cfg.work_dir = str(work)
    cfg.checkpoint_config = dict(interval=1, save_optimizer=False)
    cfg.log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook")])
    cfg.evaluation = dict(interval=args["epochs"] + 1)
    imports = dict(cfg.get("custom_imports", {}))
    imports["imports"] = list(imports.get("imports", [])) + ["training_audit"]
    imports["allow_failed_imports"] = False
    cfg.custom_imports = imports
    cfg.custom_hooks = list(cfg.get("custom_hooks", [])) + [dict(type="AgentTrainingAuditHook", mode=mode, priority="VERY_LOW")]
    cfg.dump(str(work / "training_config.py"))
    (work / "data_preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

if __name__ == "__main__":
    prepare(json.loads(Path(sys.argv[1]).read_text()))
