"""Tests for Priority 1 (parking loophole fix + 07 behavior) and Priority 2 context."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ADV_DIR = Path(__file__).resolve().parents[1] / "mcp_servers/scene_editing/core/adversarial"
if str(ADV_DIR) not in sys.path:
    sys.path.insert(0, str(ADV_DIR))

import acceptance_policy as policy
from agent.context import summarize
from mcp_servers.scene_editing.adapter.schemas import AttackInput
from mcp_servers.worldengine.adapter.policy_behavior import read_pdm_subscores, summarize_policy_behavior


def test_maneuver_intent_does_not_add_fraction_or_forbid_waiting():
    base = policy.for_new_run()
    assert base["minimum_progress_fraction"] is None

    resolved, must_progress = policy.resolve_ego_progress_policy(
        base,
        target_effect="自车提前减速并适当延后打方向，沿正确左转轨迹行驶，全程保持车身在可驾驶区域内。",
        allow_safe_stop=True,
        require_route_progress=None,
    )
    assert must_progress is True
    assert resolved["allow_safe_stop"] is True
    assert resolved["minimum_progress_fraction"] is None

    times = np.linspace(0.0, 7.5, 16)
    rec_xy = np.column_stack([times * 5.0, np.zeros_like(times)])  # 37.5m moving recorded ego
    # Short stop trajectory (13m, stopped after t=3.0s)
    short_t = np.clip(times, 0.0, 2.6)
    short_xy = np.column_stack([short_t * 5.0, np.zeros_like(times)])
    ok_short, reason_short = policy.task_progress(times, short_xy, 37.5, resolved, baseline_stopped=False)
    assert ok_short and reason_short == 'progress_diagnostic_only'

    # Neither a low ratio nor a terminal wait establishes task completion.
    half_stop_t = np.clip(times, 0.0, 4.5)
    half_stop_xy = np.column_stack([half_stop_t * 5.0, np.zeros_like(times)])  # 22.5m >= 50%, stopped 4.5..7.5s
    ok_stop, reason_stop = policy.task_progress(times, half_stop_xy, 37.5, resolved, baseline_stopped=False)
    assert ok_stop and reason_stop == 'progress_diagnostic_only'


def test_explicit_no_stop_constraint_is_preserved_without_an_implicit_fraction():
    base = policy.for_new_run()
    base["allow_safe_stop"] = False
    base["minimum_progress_fraction"] = None
    times = np.linspace(0.0, 7.5, 16)
    short_xy = np.column_stack([np.clip(times, 0.0, 2.0) * 5.0, np.zeros_like(times)])
    ok, reason = policy.task_progress(times, short_xy, 37.5, base, baseline_stopped=False)
    assert not ok and reason == 'terminal_stop_disallowed'


def test_deliberate_safe_stop_task_retains_safe_stop_exemption():
    base = policy.for_new_run()
    resolved, must_progress = policy.resolve_ego_progress_policy(
        base,
        target_effect="Brake and stop safely behind the lead vehicle.",
        allow_safe_stop=True,
        require_route_progress=None,
    )
    assert must_progress is False
    assert resolved["allow_safe_stop"] is True


def test_policy_behavior_summary_detects_early_stop_with_pdm_1(tmp_path):
    result_dir = tmp_path / "sim"
    openscene = result_dir / "WE_output/openscene_format"
    meta_dir = openscene / "meta_datas"
    meta_dir.mkdir(parents=True)

    token = "2021.10.11.07.47.13_veh-50_00736_00843-189b60c7ebb15e52"
    csv_path = openscene / "all_scenes_pdm_averages_NR.csv"
    csv_path.write_text(
        "token,step,no_at_fault_collisions,drivable_area_compliance,ego_progress,time_to_collision_within_bound,comfort,driving_direction_compliance,score,first_violation_step\n"
        f"{token},15.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,\n"
        "overall_average,15.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,\n"
    )

    # Policy drives from step 0..6 then parks at (x=13m, y=0) with minimal turn
    n = 16
    frames = []
    for i in range(n):
        x = min(float(i), 6.0) * 2.5
        pose = np.eye(4)
        pose[0, 3] = x
        frames.append({
            "token": f"f{i}",
            "log_name": token,
            "scene_token": "189b60c7ebb15e52",
            "frame_idx": i,
            "timestamp": 1_000_000 + i * 500_000,
            "ego2global": pose,
        })
    with (meta_dir / "combined.pkl").open("wb") as stream:
        pickle.dump(frames, stream)

    # Scenario PKL: recorded ego makes a 75-degree left turn over ~35m; expert stopped early like policy
    rec_angles = np.linspace(0.0, np.radians(75.0), n)
    rec_xy = np.column_stack([np.cumsum(np.r_[0.0, np.full(n - 1, 2.5) * np.cos(rec_angles[:-1])]),
                              np.cumsum(np.r_[0.0, np.full(n - 1, 2.5) * np.sin(rec_angles[:-1])])])
    exp_xy = np.column_stack([np.minimum(np.arange(n, dtype=float), 6.0) * 2.5, np.zeros(n), np.zeros(n)])
    exp_yaw = np.zeros(n)
    scenario = {
        "id": token,
        "scene_token": "189b60c7ebb15e52",
        "object_track": {"ego": {"state": {"position": exp_xy, "heading": exp_yaw}}},
        "metadata": {
            "native_export": {
                "recorded_ego_time_s": (np.arange(n, dtype=float) * 0.5).tolist(),
                "recorded_ego_rear_axle_global": rec_xy.tolist(),
                "recorded_ego_heading_rad": rec_angles.tolist(),
            }
        },
    }
    scen_dir = tmp_path / "scen"
    scen_dir.mkdir()
    with (scen_dir / "all_scenarios.pkl").open("wb") as stream:
        pickle.dump(scenario, stream)

    subscores = read_pdm_subscores(result_dir)
    assert subscores is not None
    assert subscores["scenes"][token]["score"] == 1.0

    behavior = summarize_policy_behavior(
        result_dir,
        {
            "scenario_data_folder_path": str(scen_dir),
            "scenario_pkl_file_name": "all_scenarios.pkl",
            "num_history": 4,
            "max_frame_per_scene": 16,
        },
        pdm_subscores=subscores,
    )
    assert behavior is not None
    assert behavior["stopped_detected"] is True
    assert behavior["behavior_classification"] == "stopped_at_window_end"
    assert behavior["turn_completed"] is None
    assert behavior["maneuver_completion_status"] == 'not_assessed'
    assert behavior["expert_terminal_stopped"] is True
    assert behavior["progress_ratio_vs_recorded"] < 0.5

    view = summarize("policy_runner", {
        "result_dir": str(result_dir),
        "policy_result": "completed",
        "reports": [{"succeeded": True}],
        "pdm_subscores": subscores,
        "policy_behavior_summary": behavior,
    })
    assert "pdm_subscores" in view and view["policy_behavior_summary"]["stopped_detected"] is True

    from mcp_servers.worldengine.adapter.policy_behavior import _summarize_via_subprocess

    subproc_behavior = _summarize_via_subprocess(
        result_dir,
        {
            "scenario_data_folder_path": str(scen_dir),
            "scenario_pkl_file_name": "all_scenarios.pkl",
            "num_history": 4,
            "max_frame_per_scene": 16,
        },
        pdm_subscores=subscores,
    )
    assert subproc_behavior == behavior

    # An opposite-direction full turn must not receive an automatic completion
    # verdict just because its absolute yaw or progress ratio is large.
    for i,row in enumerate(frames):
        angle=-float(rec_angles[i])
        row['ego2global']=np.array([[np.cos(angle),-np.sin(angle),0.,i*2.5],
            [np.sin(angle),np.cos(angle),0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    with (meta_dir/'combined.pkl').open('wb') as stream:pickle.dump(frames,stream)
    opposite=summarize_policy_behavior(result_dir,{'scenario_data_folder_path':str(scen_dir),
        'scenario_pkl_file_name':'all_scenarios.pkl','num_history':4,'max_frame_per_scene':16},pdm_subscores=subscores)
    assert opposite['behavior_classification']=='moving_at_window_end'
    assert opposite['signed_yaw_change_deg']<0
    assert opposite['yaw_change_ratio_vs_recorded']==1.
    assert opposite['turn_completed'] is None and opposite['maneuver_completion_status']=='not_assessed'


@pytest.mark.parametrize('text,expected',[
    ('停车等待后继续左转通过路口',True),
    ('先停车等待，随后左转通过路口',True),
    ('左转通过路口，必要时停车等待行人',True),
    ('Stop briefly, then continue driving through the junction.',True),
    ('Stop and wait, then turn left.',True),
    ('左转后在安全位置停车',False),
    ('Turn left, then stop safely behind the lead vehicle.',False),
    ('在左转路口前停车等待',False),
    ('Brake and stop safely behind the lead vehicle.',False),
    ('沿正确左转轨迹行驶，不要停车',True),
    ('Do not stop; continue driving.',True),
])
def test_intermediate_stop_and_terminal_goal_are_distinct(text,expected):
    assert policy.target_effect_requires_maneuver(text) is expected


def test_low_progress_moving_candidate_has_no_default_gate_but_explicit_config_remains():
    times=np.linspace(0.,7.5,16)
    xy=np.column_stack([times*2.,np.zeros_like(times)])
    resolved,_=policy.resolve_ego_progress_policy(policy.for_new_run(),target_effect='Continue driving.',allow_safe_stop=False)
    assert policy.task_progress(times,xy,100.,resolved)==(True,'progress_diagnostic_only')
    explicit,_=policy.resolve_ego_progress_policy(
        policy.resolve({'minimum_progress_fraction':.1}),target_effect='Continue driving.',allow_safe_stop=False)
    assert explicit['minimum_progress_fraction']==.1
    assert policy.task_progress(times,xy,100.,explicit)==(True,'minimum_progress_met')
    explicit['minimum_progress_fraction']=.8
    assert policy.task_progress(times,xy,100.,explicit)==(False,'minimum_progress_not_met')


def test_legacy_constructor_helper_has_no_implicit_half_distance_gate():
    from run_ego_reference import progress_with_safe_stop
    times=np.linspace(0.,7.5,16)
    xy=np.column_stack([times,np.zeros_like(times)])
    assert progress_with_safe_stop(times,xy,100.)==(True,'progress_diagnostic_only')


def test_attack_input_accepts_reuse_and_behavior_revision_feedback(tmp_path):
    scene_dir = tmp_path / "scene"
    (scene_dir / "metadata").mkdir(parents=True)
    frames = [{"timestamp": 1_000_000 + i * 500_000, "token": str(i)} for i in range(3)]
    (scene_dir / "metadata/frames.json").write_text(json.dumps(frames))
    (scene_dir / "metadata/cameras.json").write_text("{}")
    (scene_dir / "binding.json").write_text("{}")
    scene_ref = {
        "scene_id": "log-anchor",
        "scene_token": "anchor",
        "sample_root": str(scene_dir),
        "dataset": "test",
        "matched_camera": "CAM_F0",
        "frame_count": 3,
        "fps": 2.0,
        "manifest_path": str(scene_dir / "binding.json"),
        "frames_path": str(scene_dir / "metadata/frames.json"),
        "cameras_path": str(scene_dir / "metadata/cameras.json"),
        "cameras": {"CAM_F0": {"video_path": None, "frame_count": 3, "available_images": 0, "images_complete": False}},
        "images_complete": False,
    }
    validated = AttackInput.model_validate({
        "scene_ref": scene_ref,
        "instruction": "Cut in ahead during left turn.",
        "operation": "edit_existing",
        "reuse_completed_from": str(tmp_path / "prior_attack"),
        "revision_feedback": {
            "stage": "policy_runner",
            "outcome": "policy_stopped_safely_without_completing_maneuver",
            "failure_goal_met": False,
            "evidence": {"stopped_detected": True, "progress_ratio_vs_recorded": 0.25},
            "guidance": "Delay encroachment until ego enters the turn instead of forcing an early stop.",
        },
    })
    assert validated.reuse_completed_from == str(tmp_path / "prior_attack")
