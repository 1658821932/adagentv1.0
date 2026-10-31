"""WorldEngine command construction and two-process supervision."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def build_commands(args, environ):
    env = dict(environ)
    root = Path(env["WORLDENGINE_ROOT"]).expanduser().resolve()
    project = Path(__file__).resolve().parents[3]
    result = Path(env["WORLDENGINE_TEST_PATH"]).expanduser().resolve()
    simroot, algroot = root / "projects/SimEngine", root / "projects/AlgEngine"
    def interpreter(variable, default):
        path = Path(env.get(variable) or default).expanduser()
        return os.path.abspath(path if path.is_absolute() else project / path)
    sim_python = interpreter("SIMENGINE_PYTHON", "environments/simengine/.conda/bin/python")
    alg_python = interpreter("ALGENGINE_PYTHON", "environments/algengine/.conda/bin/python")
    folder = Path(env.get("SCENARIO_DATA_FOLDER_PATH") or f"data/sim_engine/scenarios/original/{args.data_type}").expanduser()
    if not folder.is_absolute():
        folder = root / folder
    scenario_file = folder.resolve() / env.get("SCENARIO_PKL_FILE_NAME", "all_scenarios.pkl")
    assets = Path(env.get("SCENARIO_ASSET_FOLDER_PATH") or root / "data/sim_engine/assets" / args.asset_name / "assets").expanduser().resolve()
    history = int(env.get("NUM_HISTORY", "4"))
    frames = int(env.get("MAX_FRAME_PER_SCENE", str(history + int(env.get("NUM_FUTURE", "36")))))
    future = int(env.get("NUM_FUTURE", str(frames - history)))
    if history < 1 or future < 1 or history + future != frames:
        raise ValueError("num_history + num_future must equal max_frame_per_scene")
    for unsupported in ("PLANNER_INTERVAL_FRAMES", "PLANNER_HISTORY_STRIDE", "MULTI_AGENT_VADV2"):
        if env.get(unsupported):
            raise ValueError(f"{unsupported} is not supported by the native runner")
    options = {
        "data_file_folder_path": str(folder.resolve()), "data_file_path": str(scenario_file),
        "data_pkl_file_name": scenario_file.name, "asset_folder_path": str(assets),
        "output_dir": str(result / "WE_output"), "hydra.run.dir": str(result / "hydra"),
        "job_name": f"{args.data_type}_{args.react_type}_{args.model_name}",
        "use_planner_actions": "true", "save_data": "true", "ego_controller": env.get("EGO_CONTROLLER", "log_play_controller"),
        "ego_policy": env.get("EGO_POLICY", "env_input_policy"),
        "ego_client": env.get("EGO_CLIENT", "navformer_client"),
        "ego_navigation": env.get("EGO_NAVIGATION", "trajectory_navigation"),
        "planner_data_path": str(result / "plan_traj"), "planner_client_folder": str(result / "frames"),
        "with_metric_manager": "true", "with_dense_reward_manager": env.get("WITH_DENSE_REWARD_MANAGER", "false"),
        "visualize_video": env.get("VISUALIZE_VIDEO", "false"), "visualize_BEV": env.get("VISUALIZE_BEV", "false"),
        "distributed_mode": "SINGLE_NODE", "enable_resume": "false",
        "completed_scenarios_dir": str(result / "completed_scenarios"),
        "num_history": history, "num_future": future, "sampling_poses": int(env.get("SAMPLING_POSES", "8")),
    }
    for key in ("agent_policy", "agent_navigation", "agent_client"):
        if env.get(key.upper()):
            options[("+" if key == "agent_client" else "") + key] = env[key.upper()]
        elif args.react_type == "R" and key != "agent_client":
            options[key] = "idm_policy" if key == "agent_policy" else "idm_navigation"
    if env.get("NUM_SCENARIOS"):
        options["+num_scenarios"] = int(env["NUM_SCENARIOS"])
    sim = [sim_python, "worldengine/runner/run_simulation.py", *[f"{k}={json.dumps(v) if isinstance(v,str) and (' ' in v or '[' in v or ',' in v) else v}" for k,v in options.items()]]
    planner_options = {
        "sim.monitored_folder": str(result / "frames"), "sim.plan_save_path": str(result / "plan_traj"),
        "sim.merged_ann_save_dir": str(result / "merged_ann_files"), "sim.clean_temp_files": False,
        "sim.clean_record_data": False, "sim.max_frame_per_scene": frames,
        "sim.maximum_step": frames, "queue_length": history, "model.queue_length": history,
        "data_root": str(result / "WE_output/openscene_format") + "/",
    }
    alg = [alg_python, "closed_loop/sim_test.py", args.config, args.checkpoint,
           "--log-dir", str(result / "WE_output"), "--cfg-options",
           *[f"{k}={v}" for k,v in planner_options.items()]]
    env["PYTHONPATH"] = os.pathsep.join([str(simroot), str(algroot), str(project)])
    env["PYTHONUNBUFFERED"] = "1"
    effective = {"scenario_file": str(scenario_file), "num_history": history, "num_future": future,
                 "max_frame_per_scene": frames, "simulator": options, "planner": planner_options,
                 "commands": {"simulator": sim, "planner": alg}}
    return sim, alg, simroot, algroot, result, env, effective


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "checkpoint", "model_name", "data_type", "react_type", "asset_name", "added_name"):
        parser.add_argument(name)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sim, alg, simroot, algroot, result, env, effective = build_commands(args, os.environ)
    result.mkdir(parents=True, exist_ok=True)
    (result / "effective_config.json").write_text(json.dumps(effective, indent=2))
    if args.dry_run:
        print(json.dumps(effective))
        return
    for name in ("plan_traj", "frames", "merged_ann_files", "WE_output"):
        (result / name).mkdir(exist_ok=True)
    children = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt("WorldEngine job interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        children.append(subprocess.Popen(sim, cwd=simroot, env=env))
        children.append(subprocess.Popen(alg, cwd=algroot, env=env))
        while True:
            sim_rc, alg_rc = [p.poll() for p in children]
            if sim_rc not in (None, 0) or alg_rc not in (None, 0):
                raise RuntimeError(f"WorldEngine subprocess failure: simulator={sim_rc}, planner={alg_rc}")
            if alg_rc == 0 and sim_rc is None:
                # The planner exits on the completion flag; CUDA/render cleanup
                # can keep the simulator process alive briefly after that flag.
                if not (result / "WE_output/simulation_completed.flag").is_file():
                    raise RuntimeError("Planner exited before simulator completion")
            if sim_rc == 0 and alg_rc == 0:
                return
            time.sleep(0.1)
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


if __name__ == "__main__":
    main()
