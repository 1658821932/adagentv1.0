#!/usr/bin/env python3
"""生成前端「示例」用的演示时间线：九步，内容前后一致。

①→⑧ 为构造的剧本（同一场景、同一批路径，链条自洽）；
⑨ 的数值等真实训练跑完后用 --trainer-json 覆盖。
"""
import json, sys
from datetime import datetime, timedelta
from urllib.parse import quote

ROOT = "/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagent"
SE = ROOT + "/mcp_servers/scene_editing/runtime"
WE = ROOT + "/mcp_servers/worldengine/runtime"
SCENE_FULL = "2021.10.05.04.03.05_veh-50_01003_01426-54c166c4ad5c5ad9"
SCENE = "54c166c4ad5c5ad9"

ATTACK_RUN = SE + "/run_20261112_081412_a7c3d9e15b04"
EXPERT_RUN = SE + "/run_20261112_081551_c4e8b27a60f1"
EVAL_RUN = WE + "/run_20261112_081932_51d0ae7c8f22"
TRAIN_RUN = WE + "/run_20261112_082614_2b47c91de035"
FROZEN = ATTACK_RUN + "/output/attack/materialized/frozen_scene"
NATIVE = ATTACK_RUN + "/output/native_attack"
EXPERT_NATIVE = EXPERT_RUN + "/output/native_expert"
RESULT_DIR = EVAL_RUN + "/output/simulation"
RL_DATA = EVAL_RUN + "/output/rlft_annotations"

# ---- 演示用真实媒体（前端 /api/video 按需转码）----
FAIL_VIDEO = ("/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/WorldEngine/experiments/"
              "closed_loop_exps/e2e_vadv2_50pct_fullR_bev/navtest_failures_R_complete/"
              "visualizations/cameras/camera_view_dd162898ecf351d4.mp4")
CLIP_SRC = (ROOT + "/data/navtrain_ep_per1/samples/54c166c4ad5c5ad9/preview/original_front_2hz.mp4")
# 同场景专家轨迹的 MTGS 重渲前视（今天补渲染的真实产物）
EXPERT_VIDEO = ("/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/WorldEngine/"
                "experiments/render_demo/54c166_expert/new-camera-CAM_F0.mp4")
SIM_VIDEO = (ROOT + "/mcp_servers/worldengine/core/projects/SimEngine/dbg/video_visualizer/"
             "nuplan_openscene_NR_expert_vadv2_50pct/camera_view_54c166c4ad5c5ad9.mp4")


def evidence(*pairs):
    return [dict(title=t, src=u) for t, u in pairs]


def video_url(path):
    return "/api/video?path=" + quote(path, safe="")


def eightcam_url(path):
    return "/api/eightcam?path=" + quote(path, safe="")

t0 = datetime(2026, 11, 12, 8, 14, 12)
clock = [t0]


def stamp():
    clock[0] += timedelta(seconds=1)
    return clock[0].isoformat(timespec="seconds")


def step(index, tool, reason, data_in, data_out, llm, duration, metrics=None, evidence=None):
    meta = dict(step=index, tool=tool, status="success",
                decision=dict(timestamp=stamp(), step=index, action="call_tool",
                              tool_name=tool, reason=reason),
                duration=duration, elapsed_seconds=None, evidence=evidence or [],
                llm=dict(elapsed_seconds=llm, prompt_tokens=12000 + index * 800,
                         completion_tokens=200 + index * 40),
                input=data_in, output=dict(status="success", tool=tool, result=data_out),
                artifacts=[], metrics=metrics or [], log_tail=[])
    return meta


steps = []

# ① 失败分析 -------------------------------------------------------------
steps.append(step(
    0, "failure_analyzer",
    "先从仿真归档里读出这个场景的官方指标和图像证据，确认它到底是怎么失败的，再决定复现方向。",
    dict(token="dd162898ecf351d4", simulation="navtest_failures_R_complete"),
    dict(failure_analysis=dict(
            failure_type=["跟车未减速/追尾"],
            failure_reason="自车在接近前方近乎静止车辆时持续加速，未减速避让并发生重叠接触。",
            key_scene_factors=[
                "自车速度从 0.74 m/s 单调升到 4.82 m/s，纵向加速度全程为正，没有任何减速动作。",
                "前方目标速度长期低于 0.3 m/s，间距由 0.89 m 降到 0 m 并出现重叠。",
                "车角最小可行驶余量始终为正，说明不是出界，纯粹是没刹车。",
            ],
            official_metrics=dict(step=11.0, no_at_fault_collisions=0.0, drivable_area_compliance=1.0,
                                  ego_progress=0.0, time_to_collision_within_bound=0.0, comfort=1.0,
                                  driving_direction_compliance=1.0, score=0.0, first_violation_step=7.0),
            uncertainty="第 6 步右侧驶来的深色车未能唯一匹配，是否促成未确认。"),
        token="2021.09.16.15.47.30_veh-45_01199_01391-dd162898ecf351d4",
        simulation="e2e_vadv2_50pct_fullR_bev/navtest_failures_R_complete",
        evidence_dir=ROOT + "/mcp_servers/worldengine_failure_analysis/runtime/run_20261112_081405_7a51c0be9d33"),
    llm=2.0, duration=27.1,
    evidence=evidence(("失败证据 · 归档场景录像", eightcam_url(FAIL_VIDEO))),
    metrics=[dict(key="failure_analysis.official_metrics.score", value=0),
             dict(key="failure_analysis.official_metrics.first_violation_step", value=7)]))

# ② 场景生成 -------------------------------------------------------------
DEMO_ATTACK = dict(map_context="straight_road", ego_action="straight", ego_lane_keeping="keep",
                   other_origin="ahead_same_lane", other_action="straight", other_lane_keeping="keep")
DEMO_SCENARIO = dict(
    target_scene="白天的城市直路，自车直行并保持在本车道，同车道前方一辆车保持在本车道。",
    attack=DEMO_ATTACK,
    candidates=[
        dict(candidate_id="cand_1",
             base_video_requirement="白天的城市直路，自车前方有一辆可编辑车辆。",
             retrieval_query="白天城市直路，自车直行，同车道前方清晰",
             attack_edit_strategy="调整前车运动，使其停止在自车接近的车道上。",
             ego_input_edit_strategy=None,
             ego_correct_trajectory_strategy="自车合理接近，在制动范围内减速并保持安全间距。"),
        dict(candidate_id="cand_2",
             base_video_requirement="白天的城市直路，自车前方有可放置车辆的道路空间。",
             retrieval_query="白天城市直路，自车直行，同车道前方清晰",
             attack_edit_strategy="在前方车道增加一辆停止车辆。",
             ego_input_edit_strategy=None,
             ego_correct_trajectory_strategy="自车识别停止车辆并及时减速避让。"),
        dict(candidate_id="cand_3",
             base_video_requirement="白天的城市直路，同车道前方有一辆低速车辆。",
             retrieval_query="白天城市直路，自车直行，同车道前方清晰",
             attack_edit_strategy="调整低速前车的停止时机。",
             ego_input_edit_strategy="调整自车接近速度以构建危险场景。",
             ego_correct_trajectory_strategy="自车观察前车状态，在安全距离内制动。")])
steps.append(step(
    1, "scene_instance_generator",
    "围绕相同目标场景，提供三套配套的检索条件和编辑方案。",
    dict(failure_analysis=dict(
        violation_step=7,
        failure=dict(outcome="collision", interaction="rear_end", ego_cause="未及时对停止的前车减速。"),
        attack=DEMO_ATTACK,
        environment=dict(time="2021-09-16T15:47:30+00:00", weather="unknown", extra=""),
        verification=dict(sample_frames=[5, 7, 9], corrections=[], notes=[]))),
    DEMO_SCENARIO, llm=3.0, duration=7.8))

# ③ 视频检索：演示预览，不声称运行了真实检索 ------------------------------
steps.append(step(
    2, "video_retriever",
    "示例：展示候选 cand_1 的检索结果与预览，未运行真实检索。",
    dict(requirement=DEMO_SCENARIO["candidates"][0]["retrieval_query"],
         candidate_id="cand_1", library="default"),
    dict(video_path=CLIP_SRC, source_file=CLIP_SRC, scene_ref=None,
         candidate_id="cand_1", mode="mock", match_score=0.809787),
    evidence=evidence(("模拟检索预览", video_url(CLIP_SRC))),
    llm=2.2, duration=18.2))

# ④ 编辑规划 -------------------------------------------------------------
attack_input = dict(
    scene_ref=ROOT + "/logs/demo/step_02_video_retriever/output.json",
    operation="add_vehicle", maneuver="stationary",
    vehicle=dict(asset_path=ROOT + "/data/assets/vehicle_01.ply", size_m=[4.432, 2.033, 1.637]),
    instruction=("在自车当前车道前方 22-28 米处放置一辆静止乘用车，保持其停止并留在该车道；"
                 "自车必须在 t=7.0s 之前遇到这个阻挡，从而被迫做出制动或绕行决策。不要写死碰撞。"))

steps.append(step(
    3, "edit_planner",
    "场景底图可用，失败机制明确了需要的阻碍物，因此选择新增一辆静止车辆，并把接触时间约束到评分窗口内。",
    dict(retrieval_result=ROOT + "/logs/demo/step_02_video_retriever/output.json",
         scene_requirement="城市道路直行，同车道前方车辆近乎静止，自车未及时减速。",
         target_effect="自车提前发现静止前车，在制动距离内果断减速。",
         edit_strategy="把同车道前方车辆调整为静止并保持在车道内。",
         vehicle=dict(asset_path=ROOT + "/data/assets/vehicle_01.ply", size_m=[4.432, 2.033, 1.637])),
    dict(suitability="ready",
         reasons=["直行路段视野通畅，前方车道无遮挡。",
                  "现有车流稀疏，新增一辆静止车不会与已有目标冲突。"],
         edit_plan=dict(operation="add_vehicle", maneuver="stationary",
                        target_track=None, instruction=attack_input["instruction"]),
         attack_input=attack_input,
         target_effect="自车提前发现静止前车，在制动距离内果断减速。",
         scene_id=SCENE_FULL, scene_token=SCENE),
    llm=2.4, duration=44.2))

# ⑤ 攻击编辑 -------------------------------------------------------------
frozen_traj = ATTACK_RUN + "/output/attack/trajectory_editing/candidates/seed_000/trajectory.csv"
sim_input_attack = dict(scenario_data_folder_path=NATIVE, asset_folder_path=NATIVE + "/assets",
                        scenario_pkl_file_name="all_scenarios.pkl", frame_count=24)
steps.append(step(
    4, "attack_vehicle_editor",
    "按规划在车道内 24.6 米处放置静止车辆，编辑结果通过了全部几何与任务检查，产出可仿真的冻结场景。",
    attack_input,
    dict(status="completed", stage="attack_editing", mode="real", operation="add_vehicle",
         run_dir=ATTACK_RUN, scene_id=SCENE_FULL, scene_token=SCENE,
         target_track="synthetic_vehicle_001", selected_candidate_id="seed_000",
         frozen_scene_path=FROZEN, frames_path=FROZEN + "/metadata/frames.json",
         trajectory_path=frozen_traj, submission_path=ATTACK_RUN + "/output/attack/output.json",
         simulation_input=sim_input_attack, rendered_videos={}, ilft_training_data=None,
         annotation_changes=dict(operation="append_actor",
                                 fields=["anns.gt_boxes", "anns.gt_velocity_3d", "anns.gt_names",
                                         "anns.instance_tokens", "anns.track_tokens"],
                                 native_export_report_path=NATIVE + "/export-report.json"),
         checks=dict(reference_screen="passed",
                     checks=dict(static_target="passed", lane_keeping="passed",
                                 no_ego_contact_until_s="passed", road_compliance="passed",
                                 initial_relation="passed", target_clearance="passed"),
                     physical_avoidance="undetermined")),
    llm=4.3, duration=208.6,
    metrics=[dict(key="attack.target_track", value="synthetic_vehicle_001"),
             dict(key="attack.placement_forward_m", value=24.6)]))

# ⑥ 专家轨迹 -------------------------------------------------------------
expert_traj = EXPERT_RUN + "/output/expert/expert_trajectory.csv"
sim_input_expert = dict(scenario_data_folder_path=EXPERT_NATIVE, asset_folder_path=EXPERT_NATIVE + "/assets",
                        scenario_pkl_file_name="all_scenarios.pkl", frame_count=24)
steps.append(step(
    5, "ego_trajectory_editor",
    "攻击场景已冻结，接着为同一条车道生成一条安全的专家自车轨迹：正常接近，进入制动距离后平稳减速停住。",
    dict(attack_result=ATTACK_RUN + "/output/attack/output.json",
         target_effect="自车保持合理接近速度，进入制动距离后平稳减速，在前车后方安全停住。",
         allow_safe_stop=True, use_foci=True),
    dict(status="completed", stage="ego_editing", mode="real",
         run_dir=EXPERT_RUN, scene_id=SCENE_FULL, scene_token=SCENE,
         attack_run_dir=ATTACK_RUN, target_track="synthetic_vehicle_001",
         selected_candidate_id="foci_00", selected_source="foci",
         expert_trajectory_path=expert_traj,
         trajectory_metadata_path=EXPERT_RUN + "/output/expert/trajectory-metadata.json",
         conflict_brief_path=EXPERT_RUN + "/output/expert/conflict-brief.json",
         submission_path=EXPERT_RUN + "/output/expert/output.json",
         simulation_input=sim_input_expert, ilft_training_data=None, rendered_videos={},
         checks=dict(reference_screen="passed",
                     checks=dict(max_speed="passed", max_acceleration="passed",
                                 max_reverse_speed="passed", max_lateral_speed="passed",
                                 observation_window="passed", official_margin="passed",
                                 road_tolerance="passed", progress_fraction="passed",
                                 target_clearance="passed", interpolated_contacts="passed",
                                 direction_compliance="passed", comfort="passed",
                                 map_coverage="passed", manual_video="not_checked"),
                     physical_avoidance="passed")),
    evidence=evidence(("专家轨迹回放", "/api/wevideo?path=" + quote(EXPERT_VIDEO, safe=""))),
    llm=3.9, duration=214.5,
    metrics=[dict(key="expert.source", value="foci"),
             dict(key="expert.min_target_clearance_m", value=5.96)]))

# ⑦ 策略评测 -------------------------------------------------------------
steps.append(step(
    6, "policy_runner",
    "用专家包跑原生闭环：前 1.5 秒用专家轨迹做历史，策略接管后自己开，同时采集 8192 条候选的稠密奖励用于后训练。",
    dict(execution_mode="native", collision_target_track="synthetic_vehicle_001",
         num_history=4, max_frame_per_scene=16, with_dense_reward_manager=True,
         scenario_data_folder_path=EXPERT_NATIVE, asset_folder_path=EXPERT_NATIVE + "/assets",
         scenario_pkl_file_name="all_scenarios.pkl", config_path="<默认>", checkpoint_path="<默认>",
         model_name="expert_vadv2_50pct", data_type="nuplan_openscene"),
    dict(result_dir=RESULT_DIR, policy_result="completed",
         collision_report=dict(schema_version="native_collision.v1", status="completed",
                               scene_id=SCENE_FULL, target_track="synthetic_vehicle_001",
                               policy_takeover_step=3, policy_takeover_time_s=1.5,
                               frame_count=16, eligible_frame_count=12,
                               target_collision_detected=True, any_collision_detected=True,
                               first_target_collision_time_s=6.5, min_target_clearance_m=0.0,
                               initial_history_target_contact=False, target_contact_sample_count=3),
         evaluation_scope=dict(status="completed", scope="full_rollout_pdm",
                               standard_4_second_benchmark=False,
                               scenes=[dict(scene_id=SCENE_FULL, num_history=4,
                                            max_frame_per_scene=16, sampling_poses=12,
                                            buffer_size=13, start_step=3, end_step=15,
                                            start_time_s=1.5, end_time_s=7.5, interval_s=0.5,
                                            status="completed")]),
         rollout_dir=RESULT_DIR + "/WE_output/openscene_format",
         reward_dir=RESULT_DIR + "/WE_output/openscene_format/pdms_pkl",
         candidate_rewards_present=True,
         rlft_training_data=dict(rollout_dir=RESULT_DIR + "/WE_output/openscene_format",
                                 token_file=RL_DATA + "/tokens.json",
                                 vocab_path=ROOT + "/mcp_servers/worldengine/core/data/alg_engine/test_8192_kmeans.npy",
                                 manifest_path=RL_DATA + "/manifest.json")),
    evidence=evidence(("仿真回放", video_url(SIM_VIDEO))),
    llm=2.1, duration=352.4,
    metrics=[dict(key="pdm.score", value=0.0),
             dict(key="pdm.no_at_fault_collisions", value=0.0),
             dict(key="pdm.ego_progress", value=0.0),
             dict(key="collision.first_target_collision_time_s", value=6.5)]))

# ⑧ 失败筛选 -------------------------------------------------------------
steps.append(step(
    7, "failure_case_filter",
    "读回这次闭环的官方 PDM 分数，确认它确实是零分失败场景，可以进入训练集。",
    dict(scenario_pkl=EXPERT_NATIVE + "/all_scenarios.pkl", result_dir=RESULT_DIR),
    dict(is_failure_case=True,
         failure_cases=[dict(scene_id=SCENE_FULL, scenario_pkl=EXPERT_NATIVE + "/all_scenarios.pkl",
                             pdm_score=0.0)],
         pdm_scores_file=RESULT_DIR + "/WE_output/openscene_format/all_scenes_pdm_averages_NR.csv",
         scene_count=1,
         scene_results=[dict(scene_id=SCENE_FULL, pdm_score=0.0, is_failure_case=True,
                             selection_rule="official closed-loop PDM total score == 0")]),
    llm=1.8, duration=2.9,
    metrics=[dict(key="filter.is_failure_case", value=True),
             dict(key="filter.pdm_score", value=0.0)]))

# ⑨ 后训练（先放占位，真实跑完后覆盖）------------------------------------
steps.append(step(
    8, "policy_trainer",
    "用这次的 rollout 和候选奖励做一轮 RLFT 微调，让策略在这个薄弱点上学会更早制动。",
    dict(training_mode="rlft",
         training_data=dict(rollout_dir=RESULT_DIR + "/WE_output/openscene_format",
                            token_file=RL_DATA + "/tokens.json",
                            vocab_path=ROOT + "/mcp_servers/worldengine/core/data/alg_engine/test_8192_kmeans.npy",
                            manifest_path=RL_DATA + "/manifest.json"),
         base_checkpoint=ROOT + "/mcp_servers/worldengine/core/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth",
         epochs=1, timeout=3600),
    dict(checkpoint=TRAIN_RUN + "/output/checkpoints/iter_1.pth",
         training_mode="rlft",
         log_path=TRAIN_RUN + "/logs/training.log",
         training_audit=dict(iterations=1, final_loss=0.2841, finite_loss=True,
                             parameters_updated=True, learning_rate=2e-05,
                             loss_trace=[0.4126, 0.3518, 0.3047, 0.2841]),
         data_preparation=dict(scene_count=1, frame_count=13, candidate_count=8192,
                               reward_horizon_poses=40, vocabulary_sha256="3f7c1a...9e2b",
                               kept_scenes=[SCENE_FULL], selection="PDM total score == 0")),
    llm=2.5, duration=613.8,
    metrics=[dict(key="trainer.training_mode", value="rlft"),
             dict(key="trainer.final_loss", value=0.2841)]))

total = sum(s["duration"] + (s["llm"]["elapsed_seconds"] if s["llm"] else 0) for s in steps)

doc = dict(
    run_id="run_20261112_081412_5f1c9a3e",
    task="根据这个失败案例，为它专门造一些数据用来提升",
    status="finished", error=None,
    started_at=t0.isoformat(timespec="seconds"),
    duration_seconds=round(total, 1),
    llm_model="deepseek-flash", active=False,
    summary=dict(llm_seconds=round(sum((s["llm"] or {}).get("elapsed_seconds", 0) for s in steps), 1),
                 tool_seconds=round(sum(s["duration"] for s in steps), 1),
                 steps=len(steps)),
    thinking=dict(elapsed_seconds=6.4),
    extra_decisions=[dict(action="finish",
                          reason="失败场景已复现并完成评测、筛选与一轮后训练。",
                          llm=dict(elapsed_seconds=2.6))],
    steps=steps,
)

# 允许用真实训练结果覆盖第 ⑨ 步
if "--trainer-json" in sys.argv:
    path = sys.argv[sys.argv.index("--trainer-json") + 1]
    real = json.load(open(path))
    last = doc["steps"][-1]
    last["output"] = real["output"]
    last["duration"] = real.get("duration", last["duration"])
    if real.get("llm"):
        last["llm"] = real["llm"]
    if real.get("metrics"):
        last["metrics"] = real["metrics"]
    doc["summary"]["tool_seconds"] = round(sum(s["duration"] for s in doc["steps"]), 1)
    doc["duration_seconds"] = round(doc["summary"]["tool_seconds"] + doc["summary"]["llm_seconds"], 1)

out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "run.json"
json.dump(doc, open(out, "w"), ensure_ascii=False, indent=1)
print(f"写出 {out}: {len(steps)} 步, 工具 {doc['summary']['tool_seconds']}s, "
      f"思考 {doc['summary']['llm_seconds']}s, 合计 {doc['duration_seconds']}s")
