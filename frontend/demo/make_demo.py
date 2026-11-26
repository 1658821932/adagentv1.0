#!/usr/bin/env python3
"""生成前端「示例」用的演示时间线：8 步，匹配当前 01-08 工具契约，内容前后一致。

①→⑦ 为构造的剧本（同一场景、同一批路径，链条自洽）；
⑧ 的数值可用 --trainer-json 覆盖。
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
EVAL_RUN = WE + "/run_20261112_081932_51d0ae7c8f22"
TRAIN_RUN = WE + "/run_20261112_082614_2b47c91de035"
FROZEN = ATTACK_RUN + "/output/attack/materialized/frozen_scene"
NATIVE = ATTACK_RUN + "/output/native_attack"
RESULT_DIR = EVAL_RUN + "/output/simulation"
RL_DATA = EVAL_RUN + "/output/rlft_annotations"

# ---- 演示用真实媒体（前端 /api/video 按需转码）----
FAIL_VIDEO = ("/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/WorldEngine/experiments/"
              "closed_loop_exps/e2e_vadv2_50pct_fullR_bev/navtest_failures_R_complete/"
              "visualizations/cameras/camera_view_dd162898ecf351d4.mp4")
CLIP_SRC = (ROOT + "/data/navtrain_ep_per1/samples/54c166c4ad5c5ad9/preview/original_front_2hz.mp4")
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
    return dict(
        step=index, tool=tool, status="success",
        decision=dict(timestamp=stamp(), step=index, action="call_tool",
                      tool_name=tool, reason=reason),
        duration=duration, elapsed_seconds=None, evidence=evidence or [],
        llm=dict(elapsed_seconds=llm, prompt_tokens=12000 + index * 800,
                 completion_tokens=200 + index * 40),
        input=data_in, output=dict(status="success", tool=tool, result=data_out),
        artifacts=[], metrics=metrics or [], log_tail=[],
    )


steps = []

# ① 失败分析 -------------------------------------------------------------
DEMO_FAILURE_RESULT = dict(
    token="2021.09.16.15.47.30_veh-45_01199_01391-dd162898ecf351d4",
    violation_step=7,
    simulation="e2e_vadv2_50pct_fullR_bev/navtest_failures_R_complete",
    map_location="us-nv-las-vegas-strip",
    evidence_dir=ROOT + "/logs/tool01/demo/dd162898ecf351d4/run_demo",
    bev_image=ROOT + "/logs/tool01/demo/dd162898ecf351d4/run_demo/bev_overview.png",
    failure=dict(
        outcome="collision",
        failure_driver="collision_with_vehicle",
        target_track="veh_014",
        target_class="car",
        map_context="straight_road",
        ego_behavior="自车在城市直路上沿当前车道持续加速直行，面对前方近乎静止车辆未制动。",
        other_behavior="同车道前方乘用车低速滑行并几乎静止，保持在本车道内。",
        summary="自车在城市直路上接近前方近乎静止的乘用车时未减速避让，发生追尾碰撞。",
    ),
    kinematics=dict(
        ego=dict(
            speed_start_mps=0.74, speed_violation_mps=4.82,
            displacement_m=14.6, yaw_change_deg=1.2,
        ),
        other=dict(
            size_m=[4.432, 2.033, 1.637],
            is_stationary=True,
            speed_start_mps=0.25, speed_violation_mps=0.12,
            displacement_m=0.8, yaw_change_deg=0.4,
            tail_yaw_deg=0.1, relative_heading_at_violation_deg=0.6,
            violation_distance_m=0.0,
        ),
        trajectory_ref=ROOT + "/logs/tool01/demo/dd162898ecf351d4/run_demo/trajectories.json",
    ),
    environment=dict(time="2021-09-16T15:47:30+00:00", weather="clear", extra="白天城市直路，视野清晰"),
    verification=dict(sample_frames=[5, 7, 9], notes=[]),
)

steps.append(step(
    0, "failure_analyzer",
    "先从仿真归档里读出这个场景的官方指标、BEV 俯视图和相机证据，确认它到底是怎么失败的，再决定复现方向。",
    dict(token="dd162898ecf351d4", simulation="navtest_failures_R_complete"),
    DEMO_FAILURE_RESULT,
    llm=2.0, duration=24.5,
    evidence=evidence(("失败证据 · 归档场景录像", eightcam_url(FAIL_VIDEO))),
    metrics=[dict(key="failure.outcome", value="collision"),
             dict(key="failure.failure_driver", value="collision_with_vehicle"),
             dict(key="violation_step", value=7)]))

# ② 场景需求生成 ---------------------------------------------------------
DEMO_FAILURE_CONTEXT = dict(
    outcome="collision",
    failure_driver="collision_with_vehicle",
    map_context="straight_road",
)
DEMO_SCENARIO = dict(
    target_scene="白天的城市直路，自车沿当前车道直行接近同车道前方近乎静止的乘用车。",
    failure_context=DEMO_FAILURE_CONTEXT,
    kinematics=DEMO_FAILURE_RESULT["kinematics"],
    candidates=[
        dict(candidate_id="cand_1",
             base_video_requirement="白天的城市直路，自车前方有一辆同车道可编辑乘用车。",
             retrieval_query="白天城市直路，自车直行，同车道前方清晰",
             asset_requirement=None,
             attack_edit_strategy="调整同车道前车运动，使其减速停止在自车接近的车道上。",
             ego_input_edit_strategy=None),
        dict(candidate_id="cand_2",
             base_video_requirement="白天的城市直路，自车前方有可放置车辆的道路空间。",
             retrieval_query="白天城市直路，自车直行，前方道路空旷",
             asset_requirement=dict(vehicle_class="car", size_m=[4.432, 2.033, 1.637]),
             attack_edit_strategy="在前方同车道增加一辆静止乘用车，阻挡自车前行。",
             ego_input_edit_strategy=None),
        dict(candidate_id="cand_3",
             base_video_requirement="白天的城市直路，同车道前方有一辆低速行驶的车辆。",
             retrieval_query="白天城市直路，自车直行，同车道前方有车",
             asset_requirement=None,
             attack_edit_strategy="调整低速前车的制动与停止时机，使其滞留在前方车道。",
             ego_input_edit_strategy="微调自车初始接近速度以复现近距制动窗口。"),
    ],
)
steps.append(step(
    1, "scene_instance_generator",
    "围绕相同失败上下文与目标场景，生成三套配套的检索条件和编辑方案。",
    dict(failure_analysis=DEMO_FAILURE_RESULT),
    DEMO_SCENARIO, llm=2.8, duration=7.4))

# ③ 视频检索 -------------------------------------------------------------
steps.append(step(
    2, "video_retriever",
    "按候选 cand_2 的检索词在本地视频库中检索匹配的直路源场景。",
    dict(requirement=DEMO_SCENARIO["candidates"][1]["retrieval_query"],
         candidate_id="cand_2", library="default"),
    dict(video_path=CLIP_SRC, source_file=CLIP_SRC, scene_ref=None,
         candidate_id="cand_2", mode="mock", match_score=0.842150),
    evidence=evidence(("模拟检索预览", video_url(CLIP_SRC))),
    llm=2.1, duration=16.8))

# ④ 编辑规划 -------------------------------------------------------------
attack_input = dict(
    scene_ref=dict(scene_id="navtrain_54c166c4ad5c5ad9", scene_token=SCENE),
    operation="add_vehicle",
    maneuver="stationary",
    vehicle=dict(asset_path=ROOT + "/data/assets/vehicle_01.ply", size_m=[4.432, 2.033, 1.637]),
    plan_options=[
        dict(start_xy=[24.5, 0.2], rationale="位于自车当前直行车道前方 24.5m，留有充足制动窗口。"),
        dict(start_xy=[28.0, 0.1], rationale="备选更远停靠点，避免初始历史帧重叠。"),
    ],
    instruction=("在自车当前车道前方 22-28 米处放置一辆静止乘用车，保持其停止并留在该车道；"
                 "自车必须在 t=7.0s 之前遇到这个阻挡，从而被迫做出制动或绕行决策。不要写死碰撞。"),
)

steps.append(step(
    3, "edit_planner",
    "核对检索场景 BEV 与车道拓扑，确认道路、车道和空间均通过后输出可直接交接给工具⑤的编辑方案。",
    dict(candidate_id="cand_2",
         scenario_result=ROOT + "/logs/demo/step_01_scene_instance_generator/output.json",
         retrieval_result=ROOT + "/logs/demo/step_02_video_retriever/output.json"),
    dict(suitability="ready",
         reasons=["检索底图为白天城市直路，前方同车道空间开阔，满足放置静止乘用车条件。"],
         checks=dict(road="pass", lanes="pass", space="pass"),
         target_window_s=[3.5, 7.0],
         adapted_plan=dict(
             conflict_role="stationary",
             attack_edit_strategy="在自车前方同车道 24.5m 处新增一辆静止乘用车，迫使策略及时减速或绕行。",
             attack_plans=attack_input["plan_options"],
             ego_input_edit_strategy=None,
         ),
         retrieval_suggestion=None,
         scene_id="navtrain_54c166c4ad5c5ad9",
         scene_token=SCENE,
         candidate_id="cand_2",
         target_scene=DEMO_SCENARIO["target_scene"],
         failure_context=DEMO_FAILURE_CONTEXT,
         retrieved_bev_image=ROOT + "/logs/demo/step_03_edit_planner/retrieved_bev.png",
         attack_input=attack_input,
         evidence=dict(verification_source="bev_and_map_review")),
    llm=2.4, duration=14.3))

# ⑤ 场景编辑 -------------------------------------------------------------
steps.append(step(
    4, "attack_vehicle_editor",
    "按编辑规划新增静止乘用车并完成轨迹几何校验与场景冻结，直接导出可供闭环评测的原生场景。",
    attack_input,
    dict(status="completed",
         selected_candidate_id="seed_000",
         frozen_scene_dir=FROZEN,
         native_output_dir=NATIVE,
         native_scenario_pkl=NATIVE + "/all_scenarios.pkl",
         scene_id=SCENE_FULL,
         run_dir=ATTACK_RUN,
         summary="已按 add_vehicle 方案新增静止乘用车并导出冻结与原生场景。"),
    llm=2.0, duration=48.6,
    metrics=[dict(key="editor.selected_candidate_id", value="seed_000"),
             dict(key="editor.operation", value="add_vehicle")]))

# ⑥ 策略评测 -------------------------------------------------------------
steps.append(step(
    5, "policy_runner",
    "用当前 VADv2 策略在编辑后的原生场景上跑闭环仿真，采集官方 PDM 指标与 8192 候选轨迹奖励。",
    dict(scenario_pkl=NATIVE + "/all_scenarios.pkl",
         checkpoint=ROOT + "/mcp_servers/worldengine/core/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth",
         save_rlft_annotations=True),
    dict(result_dir=RESULT_DIR,
         videos=[SIM_VIDEO],
         metrics=dict(score=0.0, no_at_fault_collisions=0.0, drivable_area_compliance=1.0,
                      ego_progress=0.0, time_to_collision_within_bound=0.0, comfort=1.0,
                      driving_direction_compliance=1.0),
         collision_report=dict(eligible_frame_count=12,
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

# ⑦ 失败案例筛选 ---------------------------------------------------------
steps.append(step(
    6, "failure_case_filter",
    "读回这次闭环的官方 PDM 分数，确认它确实是零分失败场景，可以进入后训练集。",
    dict(scenario_pkl=NATIVE + "/all_scenarios.pkl", result_dir=RESULT_DIR),
    dict(is_failure_case=True,
         failure_cases=[dict(scene_id=SCENE_FULL, scenario_pkl=NATIVE + "/all_scenarios.pkl",
                             pdm_score=0.0)],
         pdm_scores_file=RESULT_DIR + "/WE_output/openscene_format/all_scenes_pdm_averages_NR.csv",
         scene_count=1,
         scene_results=[dict(scene_id=SCENE_FULL, pdm_score=0.0, is_failure_case=True,
                             selection_rule="official closed-loop PDM total score == 0")]),
    llm=1.8, duration=2.9,
    metrics=[dict(key="filter.is_failure_case", value=True),
             dict(key="filter.pdm_score", value=0.0)]))

# ⑧ 后训练（RLFT + Reverse KL 锚定）--------------------------------------
steps.append(step(
    7, "policy_trainer",
    "用这次的 rollout、候选奖励与 Reverse KL 锚定做一轮 RLFT 微调，在不遗忘通用能力的前提下提升制动响应。",
    dict(training_mode="rlft",
         training_data=dict(rollout_dir=RESULT_DIR + "/WE_output/openscene_format",
                            token_file=RL_DATA + "/tokens.json",
                            vocab_path=ROOT + "/mcp_servers/worldengine/core/data/alg_engine/test_8192_kmeans.npy",
                            manifest_path=RL_DATA + "/manifest.json"),
         base_checkpoint=ROOT + "/mcp_servers/worldengine/core/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth",
         epochs=1, kl_weight=1.7351, max_pdm_score=1.0, timeout=3600),
    dict(checkpoint=TRAIN_RUN + "/output/checkpoints/iter_1.pth",
         training_mode="rlft",
         log_path=TRAIN_RUN + "/logs/training.log",
         training_audit=dict(iterations=1, final_loss=0.2841, finite_loss=True,
                             parameters_updated=True, learning_rate=2e-05,
                             kl_weight=1.7351,
                             loss_trace=[0.4126, 0.3518, 0.3047, 0.2841]),
         data_preparation=dict(scene_count=1, frame_count=13, candidate_count=8192,
                               reward_horizon_poses=40, vocabulary_sha256="3f7c1a...9e2b",
                               kept_scenes=[SCENE_FULL], selection="PDM total score < 1.0")),
    llm=2.5, duration=613.8,
    metrics=[dict(key="trainer.training_mode", value="rlft"),
             dict(key="trainer.kl_weight", value=1.7351),
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
                          reason="失败场景已复现并完成闭环评测、筛选与一轮 Reverse KL RLFT 后训练。",
                          llm=dict(elapsed_seconds=2.6))],
    steps=steps,
)

# 允许用真实训练结果覆盖第 ⑧ 步
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
ts = doc["summary"]["tool_seconds"]
ls = doc["summary"]["llm_seconds"]
ds = doc["duration_seconds"]
print(f"写出 {out}: {len(steps)} 步, 工具 {ts}s, 思考 {ls}s, 合计 {ds}s")
