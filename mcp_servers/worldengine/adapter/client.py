"""Agent adapter for the WorldEngine closed-loop runner."""
from __future__ import annotations

from mcp_servers.runtime import runtime_task, current_run
import json
import subprocess
from mcp_servers.process import run_logged
from pathlib import Path


from .config import PROJECT_ROOT, AGENT_PYTHON, ALGENGINE_PYTHON, WorldEngineConfig, resolve_policy_inputs, runtime_env
from common.frame_budget import resolve_frame_count, scene_frame_count
from .evaluation_horizon import DEFAULT_NATIVE_HISTORY, DEFAULT_NATIVE_FRAMES
from .policy_behavior import read_pdm_subscores, summarize_policy_behavior

REWARD_LOOKAHEAD_FRAMES = 8


@runtime_task("worldengine", "policy")
def run_policy(input_data: dict) -> dict:
    # Normal native evaluation and RL collection share one window, regardless of
    # source length. Explicit user-requested diagnostic budgets remain supported.
    input_data = dict(input_data)
    collision_only = input_data.get('collision_only', False)
    if collision_only:
        if not input_data.get('collision_target_track') or input_data.get('with_dense_reward_manager'):
            raise ValueError('Collision-only probes require a target and no dense rewards')
        input_data.update(visualize_video=False, visualize_BEV=False, with_dense_reward_manager=False)
    input_data.setdefault('num_history', DEFAULT_NATIVE_HISTORY)
    if input_data.get('max_frame_per_scene') is None and input_data.get('num_future') is None:
        input_data['max_frame_per_scene'] = DEFAULT_NATIVE_FRAMES
    frame_count = resolve_frame_count(input_data)
    # Dense reward scores candidates over a fixed 8-frame lookahead that starts at the
    # last rollout step, so the source must still hold those frames. Fail with the real
    # numbers instead of letting WorldEngine silently skip the scenario.
    if input_data.get('with_dense_reward_manager'):
        required = frame_count + REWARD_LOOKAHEAD_FRAMES
        source = None
        for key in ('scenario_data_folder_path', 'scene_dir', 'frozen_scene_path'):
            source = scene_frame_count(input_data.get(key))
            if source:
                break
        if source and source < required:
            raise ValueError(
                f"dense reward needs {required} source frames ({frame_count} rollout + "
                f"{REWARD_LOOKAHEAD_FRAMES} lookahead) but the scene has {source}; "
                f"lower max_frame_per_scene to {source - REWARD_LOOKAHEAD_FRAMES} or disable the reward manager")
    input_data = resolve_policy_inputs(input_data)
    if not AGENT_PYTHON.is_file():
        raise RuntimeError("Agent environment is missing: environments/agent/.venv/bin/python")

    import time as _time
    started_at = _time.time()

    proc = run_logged(
        [str(AGENT_PYTHON), "-m", "mcp_servers.worldengine.adapter.runner"],
        input=json.dumps({**input_data, "max_frame_per_scene": frame_count,
                          # 默认让 WE 在仿真时直接写相机视频，省掉事后合成
                          "visualize_video": input_data.get("visualize_video", True),
                          "visualize_BEV": input_data.get("visualize_BEV", False)}, ensure_ascii=False),
        cwd=PROJECT_ROOT,
        env=runtime_env(current_run("worldengine").child_env()),
        log_path=current_run("worldengine").logs / "policy.log",
        json_stdout=True,
        timeout=int(input_data.get("timeout", 86400)),
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(proc.stderr.strip() or "policy_runner returned invalid JSON") from exc
    if proc.returncode != 0 or payload.get("status") != "success":
        error = payload.get("error") or {}
        raise RuntimeError(error.get("message") or proc.stderr.strip() or "WorldEngine policy run failed")
    result = dict(payload["result"])
    reports = result.get("reports") or []
    report_entries = [entry for report in reports for entry in (report if isinstance(report, list) else [report])]
    if not report_entries or not all(isinstance(entry, dict) and entry.get("succeeded") is True for entry in report_entries):
        raise RuntimeError("WorldEngine runner report contains a failed scenario")
    result["policy_result"] = "completed"
    if input_data.get('collision_target_track'):
        report=Path(result['result_dir'])/'WE_output/native_collision_report.json'
        if not report.is_file():
            raise RuntimeError('Native run has no requested actual collision report: '+str(report))
        collision=json.loads(report.read_text())
        if collision.get('target_track')!=input_data['collision_target_track']:
            raise RuntimeError('Native collision report belongs to a different attack target')
        if collision['frame_count']<frame_count and collision.get('target_collision_detected') is not True:
            collision.update(status='incomplete_rollout',target_collision_detected=None)
        result['collision_report']=collision
    rollout = Path(result["result_dir"]) / "WE_output/openscene_format"
    if rollout.is_dir():
        result["rollout_dir"] = str(rollout)
    pdm_subscores = None if collision_only else read_pdm_subscores(result["result_dir"])
    if pdm_subscores is not None:
        result["pdm_subscores"] = pdm_subscores
    policy_behavior = summarize_policy_behavior(
        result["result_dir"],
        {**input_data, "max_frame_per_scene": frame_count},
        pdm_subscores=pdm_subscores,
    )
    if policy_behavior is not None:
        result["policy_behavior_summary"] = policy_behavior
    if input_data.get("with_dense_reward_manager", False):
        # The native planner owns this CSV; export its actual selections, never synthesize them.
        import shutil
        plan_csv = Path(result["result_dir"]) / "plan_traj/plan_idx.csv"
        reward_dir = rollout / "pdms_pkl"
        if not list(reward_dir.glob("*_scores.pkl")):
            raise RuntimeError("RL rollout completed without candidate reward files")
        if not plan_csv.is_file():
            raise RuntimeError("RL rollout is missing the actual plan_idx.csv")
        target = reward_dir / "plan_idx.csv"
        if target.resolve() != plan_csv.resolve():
            shutil.copy2(plan_csv, target)
        result["reward_dir"] = str(reward_dir)
        result["candidate_rewards_present"] = True
        # Export only actual simulation labels; the helper runs in AlgEngine's
        # NumPy/MMCV environment so its annotation pickle is training-compatible.
        task=current_run("worldengine")
        world=WorldEngineConfig().project_root
        request=dict(rollout_dir=str(rollout),vocab_path=str(world/"data/alg_engine/test_8192_kmeans.npy"),
                     output_dir=str(task.output/"rlft_annotations"),
                     config_path=input_data["config_path"],checkpoint_path=input_data["checkpoint_path"])
        request_path=task.input/"rlft_annotations_request.json"
        request_path.write_text(json.dumps(request,ensure_ascii=False,indent=2))
        env=runtime_env(task.child_env());env["WORLDENGINE_ROOT"]=str(world)
        completed=run_logged(
            [str(ALGENGINE_PYTHON),str(Path(__file__).with_name("rl_annotations.py")),
             "--request",str(request_path)],cwd=PROJECT_ROOT,env=env,
            log_path=task.logs/"rlft_annotations.log",json_stdout=True,
            timeout=int(input_data.get("timeout",86400)))
        if completed.returncode:
            raise RuntimeError("RL annotation export failed; inspect "+str(task.logs/"rlft_annotations.log"))
        result.update(json.loads(completed.stdout))
    from .policy_videos import collect_policy_videos
    videos = {} if collision_only else collect_policy_videos(result['result_dir'],WorldEngineConfig().project_root,started_at)
    if videos:
        result["videos"] = videos
    return result
