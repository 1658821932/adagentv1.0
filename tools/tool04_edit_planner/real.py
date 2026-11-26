"""Tool 04: Dual-BEV imitation planner with 3DGS recorded-trajectory constraint and zero fallback."""
from __future__ import annotations

import base64
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from mcp_servers.scene_editing.adapter.schemas import AttackInput
from tools.llm_runtime import parse_json_object, validation_details
from tools.tool01_failure_analyzer.visual_check import _yaw_text
from common.baseline_sim import read_baseline_pdm_score, resolve_baseline_frames_path
from common.scene_pool import get_pdm_failure_threshold
from .schemas import Input, Output

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_SCRIPT = PROJECT_ROOT / "tools/tool01_failure_analyzer/structure_facts.py"
RECORDS_ROOT = PROJECT_ROOT / "logs/tool04_edit_planner"
EVAL_WINDOW_S = (1.5, 7.5)
GIS_TIMEOUT_S = 180


def _gis_python() -> str | None:
    for rel in ("environments/simengine/bin/python", "environments/algengine/bin/python"):
        cand = PROJECT_ROOT / rel
        if cand.is_file():
            return str(cand)
    return None


def _render_retrieved_bev(frames_path: Path, run_dir: Path, sample_root: Path | None = None) -> dict:
    interpreter = _gis_python()
    if interpreter is None:
        return {"ok": False, "reason": "GIS/simulator Python interpreter is not available to render retrieved BEV."}
    out_bev = run_dir / "retrieved_bev.png"
    out_traj = run_dir / "retrieved_trajectories.json"
    try:
        completed = subprocess.run(
            [
                interpreter, str(STRUCTURE_SCRIPT),
                "--frames-json", str(frames_path),
                "--out-bev", str(out_bev),
                "--out-traj", str(out_traj),
                "--margin-m", "35.0",
                "--max-frames", "16",
                "--highlight-placeable-lanes",
                *(["--sample-root", str(sample_root)] if sample_root is not None else []),
            ],
            capture_output=True, text=True, timeout=GIS_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "reason": f"retrieved BEV renderer failed ({type(error).__name__})"}
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = (completed.stderr or completed.stdout or "").strip()[-300:]
        return {"ok": False, "reason": f"retrieved BEV renderer exited {completed.returncode}: {detail}"}
    try:
        payload = json.loads(completed.stdout)
    except ValueError:
        return {"ok": False, "reason": "retrieved BEV renderer produced invalid JSON"}
    if payload.get("status") != "ok":
        return {"ok": False, "reason": str(payload.get("reason") or "retrieved BEV rendering failed")}
    return {"ok": True, **payload}


def _data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _build_dual_bev_prompt(data: Input, rendered: dict, ego_traj_source: str = "policy_simulation") -> str:
    scenario = data.scenario_result
    cand = data.candidate
    fc = scenario.failure_context
    kin = scenario.kinematics
    ek1 = kin["ego"]
    ok1 = kin.get("other")
    ek2 = rendered["ego_kinematics"]

    ego1_txt = (
        f"速度{ek1['speed_start_mps']:.1f}→{ek1['speed_violation_mps']:.1f}m/s，"
        f"位移{ek1['displacement_m']:.1f}m，全程{_yaw_text(ek1['yaw_change_deg'])}，"
        f"末段1秒{_yaw_text(ek1.get('tail_yaw_deg', 0.0))}"
    )
    if isinstance(ok1, dict):
        oth1_txt = (
            f"长{ok1['size_m'][0]:.1f}m，{'静止' if ok1.get('is_stationary') else ''}"
            + ("" if ok1.get("is_stationary") else (
                f"速度{ok1['speed_start_mps']:.1f}→{ok1['speed_violation_mps']:.1f}m/s，"
                f"全程{_yaw_text(ok1['yaw_change_deg'])}，末段1秒{_yaw_text(ok1.get('tail_yaw_deg', 0.0))}"
            ))
        )
    else:
        oth1_txt = "无目标车（单车/无车失败）"

    ego2_label = "当前策略闭环仿真轨迹" if ego_traj_source == "policy_simulation" else "原采集回放轨迹"
    ego2_txt = (
        f"0..{rendered['eval_duration_s']}s（共 {rendered['eval_frames']} 帧，{ego2_label}）："
        f"速度{ek2['speed_start_mps']:.1f}→{ek2['speed_violation_mps']:.1f}m/s，"
        f"位移{ek2['displacement_m']:.1f}m，全程{_yaw_text(ek2['yaw_change_deg'])}，"
        f"末段1秒{_yaw_text(ek2.get('tail_yaw_deg', 0.0))}"
    )
    ahead = rendered.get("nearby_vehicles") or []
    behind_cnt = int(rendered.get("behind_only_count") or 0)
    if ahead:
        actor_lines = []
        for v in ahead[:6]:
            vk = v.get("kinematics") or {}
            sz = v.get("size_m") or [4.5, 2.0, 1.6]
            v0, v1 = vk.get("speed_start_mps", 0.0), vk.get("speed_violation_mps", 0.0)
            mtxt = "静止" if vk.get("is_stationary") else f"速度{v0:.1f}→{v1:.1f}m/s"
            actor_lines.append(f"  - {v['vehicle']} (track={v['track'][:8]}): 长{sz[0]:.1f}m，最近间距{v['min_distance_m']}m，{mtxt}")
        actors2_txt = "\n" + "\n".join(actor_lines) + (f"\n  （另有{behind_cnt}辆车位于起点后方，不占前方车道）" if behind_cnt else "")
    else:
        actors2_txt = f"无（0辆；另有{behind_cnt}辆车均位于自车起点后方，不占用前方及对向车道）"

    asset_txt = "null（不新增车辆）"
    if cand.asset_requirement is not None:
        asset_txt = json.dumps({
            "reference_asset": cand.asset_requirement.model_dump(mode="json"),
            "explicit_vehicle_class": data.vehicle_catalog_class,
            "scope": "参考车型和尺寸用于理解原场景；仅显式车型是约束。实际资产由工具⑤根据道路校验选择。",
        }, ensure_ascii=False)
    return (
        "你是自动驾驶对抗场景编辑规划员（Tool04）。你会同时收到两张同款配色的 BEV 图：\n"
        "- 【图1：原危险失败场景仿真 BEV】（亮品红为前 1.5s 历史段，橙色为 1.5s 后闭环仿真轨迹，亮红粗框为首次违规帧，蓝色为原场景他车；车身每 2 帧抽样绘制、不标文字标号，黑色箭头指示车头方向）。\n"
        f"- 【图2：新检索到的底场景标准16帧 BEV（0–7.5s）】（亮品红为前 1.5s 可编辑历史输入段，橙色为 1.5–7.5s {ego2_label}，蓝色为前方/侧方既有车辆，车身每 2 帧抽样绘制、不标文字标号，黑色箭头指示车头方向；淡绿色半透明区域为 3DGS 有效重建路面覆盖区；**亮青色高亮实线（带深青方向箭头）为已校验有有效路面高程的可行驶车道中心线（箭头指示合法行驶方向）**，新增攻击车起点 S1/S2/S3 必须直接落在亮青色线上）。\n\n"
        "请对照图1的危险交互核心意图，评估图2是否大致符合该类危险场景的道路、车道与空间条件，并在图2上给出适配规划（不要求道路几何、曲率或速度曲线与图1完全一致，符合大致场景意图并可合理构造同类危险交互即可）。\n\n"
        "【原失败场景信息（图1）】\n"
        f"- 失败类型：{fc.outcome} / {fc.failure_driver}，最小道路形态：{fc.map_context}\n"
        f"- 目标场景：{scenario.target_scene}\n"
        f"- 原自车运动：{ego1_txt}\n"
        f"- 原目标车运动：{oth1_txt}\n\n"
        f"【待执行的候选方案（{cand.candidate_id}）】\n"
        f"- 底视频要求：{cand.base_video_requirement}\n"
        f"- 资产需求：{asset_txt}\n"
        f"- 粗攻击策略：{cand.attack_edit_strategy}\n"
        f"- 粗自车调整：{cand.ego_input_edit_strategy}\n\n"
        f"【新检索真实底场景信息（图2：{data.scene_ref.scene_token}）】\n"
        f"- 源场景总帧数：{rendered['frame_count']} 帧；标准评测窗：0..{rendered['eval_duration_s']}s（共 {rendered['eval_frames']} 帧，2Hz）\n"
        f"- 新场景自车运动：{ego2_txt}\n"
        "- 图2坐标网格：以自车起点(t=0)为(0,0)，主网格10m，虚线网格5m，请直接从图2网格读取坐标。\n"
        f"- 自车前方及对向区域既有车辆：{actors2_txt}\n\n"
        "【输出 JSON 格式（严格只含以下字段）】\n"
        "{\n"
        "  \"suitability\": \"ready | unsuitable\",\n"
        "  \"checks\": {\"road\": \"pass | fail\", \"lanes\": \"pass | fail\", \"space\": \"pass | fail\"},\n"
        "  \"reasons\": [\"对照图1与图2的具体判断依据\"],\n"
        "  \"target_window_s\": [3.0, 5.5],\n"
        "  \"target_vehicle_id\": \"若编辑既有车，从上方既有车辆列表中填写对应编号如V1，否则填null\",\n"
        "  \"adapted_plan\": {\n"
        "    \"conflict_role\": \"oncoming | crossing | adjacent_cut_in | lead_vehicle | stationary\",\n"
        "    \"attack_edit_strategy\": \"针对图2具体帧段与车道位置，如何模仿图1放置并控制目标车（单车/unsuitable时填null）\",\n"
        "    \"attack_plans\": [\n"
        "      {\"start_xy\": [x1, y1], \"rationale\": \"方案1：与图1同源来车支路青色中心线上的近段起点\"},\n"
        "      {\"start_xy\": [x2, y2], \"rationale\": \"方案2：与图1同源来车支路青色中心线上的中段起点\"},\n"
        "      {\"start_xy\": [x3, y3], \"rationale\": \"方案3：与图1同源来车支路青色中心线上的远段起点\"}\n"
        "    ],\n"
        "    \"ego_input_edit_strategy\": \"可选定性备注：仅作参考（是否实际编辑自车前1.5s由工具5自行决定；无需说明或unsuitable时填null）\"\n"
        "  },\n"
        "  \"retrieval_suggestion\": \"若unsuitable，给出新的精简检索词（≤48字，禁用编辑词）；若ready填null\"\n"
        "}\n\n"
        "【硬约束】\n0. 严格可编辑性门槛（先判有无对应车道，严禁硬凑）：先对照图1所需的交互车道关系（例如无物理隔离的对向相邻车道、同向相邻车道、路口可汇入支路等），核查图2自车未来轨迹（橙色 1.5–7.5s 段）附近是否真实存在对应角色的亮青色车道（深青箭头方向吻合，且与自车之间没有白色导流岛/中央隔离带物理隔断）。若图2根本缺少该类交互车道（例如需要对向相邻车道，但图2自车行驶在被导流岛隔开的单向匝道或纯同向多车道上），必须直接判 suitability=\"unsuitable\" 并给出新的 retrieval_suggestion 换底场景，严禁硬把起点塞到后方同向车道或无关支路！\n"
        "1. 3DGS 高斯重建轨迹约束：图2的高斯点云围绕底场景自车轨迹重建，自车只允许沿原路线微调速度或小幅横向偏移，绝不允许改道去未采集的支路。只要图2自车轨迹段（亮品红+橙色）大致契合图1的核心交互场景（允许曲率、速度或周边路网细节不同）即可判 ready；仅当道路形态/方向根本不符、自车处于被白色隔离带/导流岛物理隔开的单向专用匝道导致目标车无法自然切入交互、或无空间构造冲突时，才判 unsuitable。\n"
        "2. 操作与空间一致性：若候选带 asset_requirement（新增车辆），图2冲突区及上游接近车道在预期交互时段内必须空旷无前方/对向阻挡车（起点后方无关车不影响前方空间）；若候选为编辑既有车辆，必须从上方既有车辆列表中选中真实存在的 V1..Vn 填入 target_vehicle_id（图2画面本身不标车号，请结合列表与位置对应）。\n"
        "3. 先从图1确认目标车与自车的冲突角色 conflict_role：对向来车 oncoming、路口横穿 crossing、同向邻道切入 adjacent_cut_in、同车道前车 lead_vehicle、静止障碍 stationary。单车场景填 null。"
        "在图2沿对应车道箭头追踪合法流向，确认该流向能构成同类交互：对向来车要沿车道向自车驶来，横穿车要沿车道进入交汇区，同向切入车要从邻道接近，同车道前车沿自车行驶方向行驶。"
        "给出定性编辑思路，有目标车时独立读出 3 个起点，每组只写 start_xy 和 rationale，说明合法流向如何构成该交互。具体相遇点、速度和切入幅度由工具5求解，保持自然的浅度压线。\n"
        "4. 逐点读图（严禁等差数列外推）：每个 start_xy 必须独立盯着图2的 5m/10m 网格读取！所有 start_xy 必须**直接落在图2的亮青色实线上**（亮青色线即已有有效路面高程的合法车道中心线，深青箭头表示行驶方向），严禁落到没有青色高亮覆盖的空白/远端无高程段或浅灰马路外；3 个起点沿来车方向依次布置在近/中/远上游段（彼此拉开间距，且距自车初始位置（亮品红前 1.5s 段）至少 8m 以上）。\n"
        "5. 若判 ready，checks 三项必须全为 pass，有目标车时 target_window_s 给出参考冲突时段（建议落在 [3.0, 7.5] 秒内，留足接管后反应时间；该字段仅作定性参考，不锁死工具5的求解窗口）；若判 unsuitable，adapted_plan、target_window_s、target_vehicle_id 必须为 null。"
    )


def _ego_anchors_text(frames_path: Path, eval_frames: int) -> str:
    try:
        frames = json.loads(frames_path.read_text())[:eval_frames]
        ox, oy = float(frames[0]["ego2global"][0][3]), float(frames[0]["ego2global"][1][3])
        picks = [i for i in (0, 3, 6, 9, 12, eval_frames - 1) if 0 <= i < len(frames)]
        return ", ".join(
            f"E{i}=({frames[i]['ego2global'][0][3] - ox:.1f}, {frames[i]['ego2global'][1][3] - oy:.1f})"
            for i in dict.fromkeys(picks)
        )
    except Exception:
        return "未知"


def _map_xy_to_local(frames_path: Path, xy: list[float] | None) -> list[float] | None:
    """Convert E0-relative (dx_east, dy_north) from Figure 2 axes into Tool 05 E0-heading local (x_m, y_m)."""
    if not (isinstance(xy, list) and len(xy) == 2 and all(isinstance(v, (int, float)) for v in xy)):
        return None
    try:
        import math
        frames = json.loads(frames_path.read_text())
        m = frames[0]["ego2global"]
        dx, dy = float(xy[0]), float(xy[1])
        # If caller/VLM supplied absolute UTM coords by mistake, subtract E0 origin
        x0, y0 = float(m[0][3]), float(m[1][3])
        if abs(dx) > 1000.0 or abs(dy) > 1000.0:
            dx, dy = dx - x0, dy - y0
        if math.hypot(dx, dy) > 250.0:
            return None
        yaw0 = math.atan2(float(m[1][0]), float(m[0][0]))
        lx = dx * math.cos(yaw0) + dy * math.sin(yaw0)
        ly = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
        return [round(lx, 2), round(ly, 2)]
    except Exception:
        return None



def _load_road_height_checker(sample_root: Path):
    """Pure-Python road_height_map.npy checker (~0.01s, no numpy required)."""
    import ast, re, struct
    cfg_path = sample_root / "asset_config.yaml"
    rhm_dir = None
    for sub in ("source_asset/road_height_map", "asset/road_height_map", "road_height_map"):
        cand = sample_root / sub
        if (cand / "road_height_map.npy").is_file() and (cand / "sim2.json").is_file():
            rhm_dir = cand
            break
    if rhm_dir is None or not cfg_path.is_file():
        return None
    try:
        block = cfg_path.read_text().split("road_block:", 1)
        nums = re.findall(r"-\s+([-0-9.]+)", block[1])[:4] if len(block) == 2 else []
        if len(nums) < 4:
            return None
        x0, y0, x1, y1 = (float(v) for v in nums)
        recon_x, recon_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        sim2 = json.loads((rhm_dir / "sim2.json").read_text())
        R = sim2["R"]
        t = sim2["t"]
        scale = float(sim2["s"])
        raw = (rhm_dir / "road_height_map.npy").read_bytes()
        if raw[:6] != b"\x93NUMPY":
            return None
        major = raw[6]
        hlen = struct.unpack("<H", raw[8:10])[0] if major == 1 else struct.unpack("<I", raw[8:12])[0]
        hstart = 10 if major == 1 else 12
        hdr = ast.literal_eval(raw[hstart:hstart + hlen].decode("latin1"))
        shape = hdr["shape"]
        if len(shape) != 2:
            return None
        rows, cols = int(shape[0]), int(shape[1])
        descr = str(hdr["descr"])
        fmt_ch = "f" if "f4" in descr else ("d" if "f8" in descr else None)
        if fmt_ch is None:
            return None
        endian = ">" if descr.startswith(">") else "<"
        data_bytes = raw[hstart + hlen:]
        elem_sz = 4 if fmt_ch == "f" else 8

        def has_bilinear(hx: float, hy: float) -> bool:
            px = scale * (R[0][0] * hx + R[0][1] * hy + t[0])
            py = scale * (R[1][0] * hx + R[1][1] * hy + t[1])
            if not (0.0 <= px <= cols - 1 and 0.0 <= py <= rows - 1):
                return False
            ix0, iy0 = int(math.floor(px)), int(math.floor(py))
            ix1, iy1 = min(ix0 + 1, cols - 1), min(iy0 + 1, rows - 1)
            for iy in (iy0, iy1):
                row_off = iy * cols
                for ix in (ix0, ix1):
                    off = (row_off + ix) * elem_sz
                    val = struct.unpack_from(endian + fmt_ch, data_bytes, off)[0]
                    if not math.isfinite(val):
                        return False
            return True

        def footprint_ok(wx: float, wy: float) -> bool:
            # Align with Tool 05: only require the vehicle center point to have valid road height
            hx, hy = wx - recon_x, wy - recon_y
            return has_bilinear(hx, hy)

        return footprint_ok
    except Exception:
        return None


def _nearest_segment(point, segments):
    best = (math.inf, None, None)
    x, y = point
    for (ax, ay), (bx, by) in segments:
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        s = min(length, max(0.0, (x - ax) * ux + (y - ay) * uy))
        px, py = ax + s * ux, ay + s * uy
        distance = math.hypot(x - px, y - py)
        if distance < best[0]:
            best = (distance, (px, py), (ux, uy))
    return best


def _check_attack_plans_geometry(frames_path: Path, plans: list[dict], rendered: dict, role: str, scene_dir: Path | None = None):
    """Validate E0-relative starts against the directed centerlines drawn in Figure 2."""
    frames = json.loads(frames_path.read_text())
    e2g0 = frames[0]["ego2global"]
    ox, oy = float(e2g0[0][3]), float(e2g0[1][3])
    rhm_ok = _load_road_height_checker(scene_dir if scene_dir is not None else frames_path.parent.parent)
    ego = rendered.get("ego_bev_xy", [])
    lanes = rendered.get("placeable_lanes", [])
    lane_segments = [list(zip(lane, lane[1:])) for lane in lanes if len(lane) >= 2]
    ego_future = list(zip(ego[3:], ego[4:]))
    if not lane_segments or len(ego) < 3 or (not ego_future and role in ("oncoming", "crossing")):
        return ["缺少图2的可放置车道或自车未来轨迹，无法核验起点。"], []

    issues, ok_points = [], []
    for idx, item in enumerate(plans, 1):
        st = item.get("start_xy") if isinstance(item, dict) else None
        if not (isinstance(st, list) and len(st) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in st
        )):
            issues.append(f"S{idx} 坐标格式无效")
            continue
        sx, sy = st
        label = f"S{idx}=[{sx:.1f}, {sy:.1f}]"
        matches = [_nearest_segment(st, segments) for segments in lane_segments]
        best_idx = min(range(len(matches)), key=lambda i: matches[i][0])
        lane_gap, point, direction = matches[best_idx]
        if lane_gap > 2.2:
            issues.append(f"{label} 距图2最近的亮青色车道中心线超过 2.2m，请沿网格重新选点。")
            continue
        if min(math.hypot(sx - ex, sy - ey) for ex, ey in ego[:3]) < 8.0:
            issues.append(f"{label} 距离自车初始位置 E0..E2 不足 8m。")
            continue
        if rhm_ok is not None and not rhm_ok(ox + sx, oy + sy):
            issues.append(f"{label} 中心无有效路面高程，请移到亮青色车道的有效段。")
            continue
        px, py = point
        ux, uy = direction
        # A coarse grid read between opposite lanes cannot establish which flow was selected.
        if any(
            gap <= min(2.2, lane_gap + 0.75) and ux * other_dir[0] + uy * other_dir[1] < -0.5
            for gap, _, other_dir in matches
        ):
            issues.append(
                f"{label} 离两条反向车道都太近，无法确认来向。"
                f"较近的中心线经过 ({px:.1f}, {py:.1f})，"
                f"流向 ({px + 5 * ux:.1f}, {py + 5 * uy:.1f})。"
                "请按冲突角色重读对应亮青色线的坐标，避免落在两条线之间。"
            )
            continue
        if role == "oncoming":
            ego_all = list(zip(ego, ego[1:]))
            if ego_all:
                _, _, ego_dir = _nearest_segment(point, ego_all)
                cos_ang = max(-1.0, min(1.0, ux * ego_dir[0] + uy * ego_dir[1]))
                ang_deg = math.degrees(math.acos(cos_ang))
                if ang_deg < 90.0:
                    issues.append(
                        f"{label} 所在车道的合法流向与自车同段方向夹角仅 {ang_deg:.0f}°（<90°，属于同向追尾车道而非对向来车）。"
                        "对向来车必须放在与自车反向迎面驶来的亮青色车道中心线上；若自车行驶在被隔离带/导流岛隔开的单向匝道、无合法对向交互车道，请改判 unsuitable。"
                    )
                    continue
        if role in ("oncoming", "crossing"):
            # Trace forward along the actual directed lane polyline up to 25m
            # so curved ramps that bend slightly outward before merging are not falsely rejected.
            segs = lane_segments[best_idx]
            best_si = min(
                range(len(segs)),
                key=lambda si: _nearest_segment(st, [segs[si]])[0],
            )
            forward_pts = []
            cur = point
            traveled = 0.0
            for si in range(best_si, len(segs)):
                nxt = segs[si][1]
                seg_d = math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
                if seg_d > 1e-6:
                    rem = 25.0 - traveled
                    if seg_d >= rem:
                        ratio = rem / seg_d
                        forward_pts.append((cur[0] + ratio * (nxt[0] - cur[0]), cur[1] + ratio * (nxt[1] - cur[1])))
                        traveled = 25.0
                        break
                    forward_pts.append(nxt)
                    traveled += seg_d
                cur = nxt
            if traveled < 5.0:
                forward_pts.append((px + 5.0 * ux, py + 5.0 * uy))
            ahead = forward_pts[0]
            behind = (px - 5.0 * ux, py - 5.0 * uy)
            gap, _, _ = _nearest_segment(point, ego_future)
            min_ahead_gap = min(_nearest_segment(pt, ego_future)[0] for pt in forward_pts)
            behind_gap, _, _ = _nearest_segment(behind, ego_future)
            if min_ahead_gap > gap + 2.0 and behind_gap < gap:
                issues.append(
                    f"{label} 所在车道的合法流向从 ({px:.1f}, {py:.1f}) "
                    f"朝 ({ahead[0]:.1f}, {ahead[1]:.1f})，沿此方向前进会持续远离自车未来轨迹。"
                    "请在图2重新选择朝交汇区行驶的车道，按网格重读起点。"
                )
                continue
        elif role in ("adjacent_cut_in", "lead_vehicle"):
            ego_all = list(zip(ego, ego[1:]))
            if ego_all:
                _, _, ego_dir = _nearest_segment(point, ego_all)
                cos_ang = max(-1.0, min(1.0, ux * ego_dir[0] + uy * ego_dir[1]))
                ang_deg = math.degrees(math.acos(cos_ang))
                if ang_deg > 60.0:
                    issues.append(
                        f"{label} 所在车道的合法流向与自车同段行驶方向夹角约 {ang_deg:.0f}°（>60°），"
                        f"疑似落在了对向或横向车道。冲突角色 {role} 请选择与自车同向行驶的亮青色车道中心线。"
                    )
                    continue
        ok_points.append(label)
    return issues, ok_points


def _render_self_check_overlay(frames_path: Path, plans: list[dict], out_path: Path, sample_root: Path | None = None) -> Path | None:
    """Plot the proposed S1..S3 on the Tool 04 BEV for visual self-correction."""
    import subprocess
    script = f"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
sys.path.insert(0, str(Path({str(Path(__file__).resolve().parents[2] / 'tools/tool01_failure_analyzer')!r})))
import structure_facts as sf

frames = json.loads(Path({str(frames_path)!r}).read_text())
ox, oy = float(frames[0]['ego2global'][0][3]), float(frames[0]['ego2global'][1][3])
plans = {json.dumps(plans, ensure_ascii=False)}
colors = ['#1b5e20', '#0d47a1', '#6a1b9a']

orig_savefig = Figure.savefig
def hooked(self, fname, *args, **kwargs):
    ax = self.axes[0]
    for i, item in enumerate(plans):
        col = colors[i % len(colors)]
        st = item.get('start_xy')
        if isinstance(st, list) and len(st) == 2:
            wx, wy = float(st[0]) + ox, float(st[1]) + oy
            ax.scatter([wx], [wy], s=220, marker='*', color=col, edgecolor='black', linewidth=0.8, zorder=20)
            ax.text(wx + 0.8, wy + 1.2, f'S{{i+1}}', fontsize=9, fontweight='bold', color=col, zorder=21,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=col, alpha=0.9))
    ax.set_title('SELF-CHECK: Your S1..S3 (stars) on Figure 2')
    return orig_savefig(self, fname, *args, **kwargs)

Figure.savefig = hooked
sys.argv = ['structure_facts.py', '--frames-json', {str(frames_path)!r}, *(['--sample-root', {str(sample_root)!r}] if {sample_root is not None!r} else []),
            '--out-bev', {str(out_path)!r}, '--out-traj', {str(out_path.with_suffix('.json'))!r},
            '--max-frames', '16', '--margin-m', '35.0', '--highlight-placeable-lanes']
sf.main()
"""
    proc = subprocess.run([str(Path(__file__).resolve().parents[2] / "environments/simengine/bin/python"), "-c", script], cwd=str(Path(__file__).resolve().parents[2]), capture_output=True, text=True, timeout=60)
    return out_path if proc.returncode == 0 and out_path.is_file() else None



def _needs_ego_history_edit(strategy: str | None) -> bool:
    if not isinstance(strategy, str) or not strategy.strip():
        return False
    txt = strategy.strip()
    no_edit_markers = ("无需", "不调整", "不修改", "不需要", "保持原", "保持自车沿原", "原采集轨迹即可", "不改")
    return not any(m in txt for m in no_edit_markers)


def _build_attack_input(data: Input, rendered: dict, decision: dict) -> dict | None:
    fc = data.scenario_result.failure_context
    if fc.failure_driver in ("departure_ego_only", "collision_non_vehicle"):
        plan = decision.get("adapted_plan") or {}
        ego_strat = plan.get("ego_input_edit_strategy") or ""
        instruction = f"{data.scenario_result.target_scene} {ego_strat}".strip()
        base_pdm = read_baseline_pdm_score(data.scene_ref.scene_token)
        need_ego_edit = True if (base_pdm is None or base_pdm >= get_pdm_failure_threshold()) else False
        request = {
            "scene_ref": data.scene_ref.model_dump(mode="json"),
            "instruction": instruction,
            "operation": "ego_only",
            "prepare_ego_history": need_ego_edit,
            "constraints": {"no_ego_contact_until_s": 0.0, "target_window_s": None},
        }
        if data.scenario_result.kinematics:
            request["reference_kinematics"] = data.scenario_result.kinematics
        if data.scenario_result.source_bev_image:
            request["reference_bev_image"] = data.scenario_result.source_bev_image
        return AttackInput.model_validate(request).model_dump(mode="json")
    cand = data.candidate
    plan = decision["adapted_plan"]
    window = decision["target_window_s"]
    other_kin = (data.scenario_result.kinematics or {}).get("other") or {}

    kin_note = ""
    if other_kin:
        motion_desc = "静止" if other_kin.get("is_stationary") else (
            "左转" if float(other_kin.get("yaw_change_deg", 0.0)) >= 25.0 else (
                "右转" if float(other_kin.get("yaw_change_deg", 0.0)) <= -25.0 else "直行/小幅变向"
            )
        )
        kin_note = f" 目标车定性运动趋势：{motion_desc}（具体速度与轨迹数值由工具5结合当前道路微调）。"
    instruction = f"{data.scenario_result.target_scene} {plan['attack_edit_strategy']}{kin_note}".strip()

    if cand.asset_requirement is not None:
        maneuver = "stationary" if other_kin.get("is_stationary") else "moving"
        request = {
            "scene_ref": data.scene_ref.model_dump(mode="json"),
            "instruction": instruction,
            "operation": "add_vehicle",
            "maneuver": maneuver,
            **({"vehicle": data.vehicle.model_dump(mode="json")} if data.vehicle is not None else
               {"vehicle_catalog_class": data.vehicle_catalog_class or "any"}),
            "prepare_ego_history": False,
            "constraints": {"no_ego_contact_until_s": 3.0, "target_window_s": None},
        }
    else:
        vid = decision.get("target_vehicle_id")
        match = next((v for v in rendered.get("nearby_vehicles") or [] if v["vehicle"] == vid), None)
        if match is None:
            raise ValueError(f"edit_existing candidate requires a valid target_vehicle_id from 图2, got {vid!r}")
        request = {
            "scene_ref": data.scene_ref.model_dump(mode="json"),
            "instruction": instruction,
            "operation": "edit_existing",
            "target_track": match["track"],
            "prepare_ego_history": False,
            "constraints": {"no_ego_contact_until_s": 3.0, "target_window_s": None},
        }
    if data.scenario_result.kinematics:
        request["reference_kinematics"] = data.scenario_result.kinematics
    if data.scenario_result.source_bev_image:
        request["reference_bev_image"] = data.scenario_result.source_bev_image
    frames_p = Path(data.scene_ref.frames_path)
    paired_local = []
    for item in plan.get("attack_plans") or []:
        st_l = _map_xy_to_local(frames_p, item.get("start_xy"))
        if st_l is not None:
            paired_local.append({
                "start_xy_m": st_l,
                "rationale": item.get("rationale", ""),
            })
    if paired_local:
        request["spatial_hint"] = {
            "attack_start_xy_m": paired_local[0]["start_xy_m"],
            "attack_start_candidates_xy_m": [p["start_xy_m"] for p in paired_local],
            "attack_plans_local": paired_local,
        }
    return AttackInput.model_validate(request).model_dump(mode="json")


def _parse_vlm_decision(raw: str, single_vehicle: bool) -> dict:
    answer = parse_json_object(raw)
    required = {"suitability", "checks", "reasons", "target_window_s", "target_vehicle_id", "adapted_plan", "retrieval_suggestion"}
    if not isinstance(answer, dict) or set(answer) != required:
        raise ValueError(f"Dual-BEV VLM response must contain exact keys {sorted(required)}")
    suit = answer["suitability"]
    if suit not in ("ready", "unsuitable"):
        raise ValueError(f"suitability must be ready or unsuitable, got {suit!r}")
    if suit == "ready":
        win = answer["target_window_s"]
        if not single_vehicle:
            if not (isinstance(win, list) and len(win) == 2 and all(isinstance(x, (int, float)) for x in win)):
                raise ValueError("ready vehicle-driven plan requires numeric target_window_s=[start, end]")
            if not (EVAL_WINDOW_S[0] <= float(win[0]) < float(win[1]) <= EVAL_WINDOW_S[1]):
                raise ValueError(f"target_window_s {win} must lie inside {list(EVAL_WINDOW_S)}")
            answer["target_window_s"] = [round(float(win[0]), 2), round(float(win[1]), 2)]
        else:
            answer["target_window_s"] = None
    else:
        answer["adapted_plan"] = None
        answer["target_window_s"] = None
        answer["target_vehicle_id"] = None
    return answer


def _evaluate_one(data: Input, run_dir: Path, llm_client=None) -> tuple[dict, str]:
    ref = data.scene_ref
    missing = []
    if data.retrieval_result.mode == "mock":
        missing.append("Mock retrieval does not establish a real source scene")
    frames_path = Path(ref.frames_path).expanduser()
    if not frames_path.is_file():
        missing.append(f"Missing scene annotations: {frames_path}")
    if not Path(ref.cameras_path).expanduser().is_file():
        missing.append(f"Missing camera metadata: {ref.cameras_path}")
    source_bev = Path(str(data.scenario_result.source_bev_image or ""))
    if not source_bev.is_file():
        missing.append(f"Missing Tool01 source BEV image: {data.scenario_result.source_bev_image!r}")
    if missing:
        return ({
            "suitability": "needs_information",
            "reasons": missing,
            "checks": {"road": "unknown", "lanes": "unknown", "space": "unknown"},
            "target_window_s": None,
            "adapted_plan": None,
            "retrieval_suggestion": None,
            "retrieved_bev_image": None,
            "attack_input": None,
            "evidence": {"missing_resources": missing},
        }, "none")

    baseline_sim_frames = resolve_baseline_frames_path(ref.scene_token)
    planning_frames_path = baseline_sim_frames or frames_path
    ego_traj_source = "policy_simulation" if baseline_sim_frames is not None else "recorded_replay"
    rendered = _render_retrieved_bev(planning_frames_path, run_dir, sample_root=frames_path.parent.parent)
    if not rendered.get("ok"):
        reason = rendered["reason"]
        return ({
            "suitability": "needs_information",
            "reasons": [reason],
            "checks": {"road": "unknown", "lanes": "unknown", "space": "unknown"},
            "target_window_s": None,
            "adapted_plan": None,
            "retrieval_suggestion": None,
            "retrieved_bev_image": None,
            "attack_input": None,
            "evidence": {"render_error": reason},
        }, "none")

    from agent.llm_client import LLMClient
    client = llm_client or LLMClient(component="edit_planner")
    prompt = _build_dual_bev_prompt(data, rendered, ego_traj_source=ego_traj_source)
    raw = client.chat([
        {"role": "system", "content": "只输出一个合法 JSON 对象，不要 Markdown 代码块，不要任何多余解释。"},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _data_url(source_bev)}},
            {"type": "image_url", "image_url": {"url": _data_url(Path(rendered["bev_image"]))}},
        ]},
    ])
    (run_dir / "vlm_response.txt").write_text(raw if isinstance(raw, str) else "")
    single_vehicle = data.scenario_result.failure_context.failure_driver in ("departure_ego_only", "collision_non_vehicle")
    decision = _parse_vlm_decision(raw, single_vehicle)

    self_check_issues = []
    if decision["suitability"] == "ready" and not single_vehicle and decision.get("adapted_plan"):
        plan = decision["adapted_plan"]
        role = plan.get("conflict_role")
        for sc_round in range(3):
            plans = plan.get("attack_plans")
            if not plans:
                if data.candidate.asset_requirement is not None:
                    self_check_issues = ["新增车辆缺少候选起点，无法核验车道流向。"]
                break
            if role not in ("oncoming", "crossing", "adjacent_cut_in", "lead_vehicle", "stationary"):
                self_check_issues = ["缺少有效的冲突角色，无法核验候选起点。"]
                break
            self_check_issues, ok_points = _check_attack_plans_geometry(planning_frames_path, plans, rendered, role, scene_dir=frames_path.parent.parent)
            if not self_check_issues or sc_round == 2:
                break
            overlay_name = "self_check_overlay.png" if sc_round == 0 else f"self_check_overlay_r{sc_round + 1}.png"
            overlay_p = _render_self_check_overlay(planning_frames_path, plans, run_dir / overlay_name, sample_root=frames_path.parent.parent)
            if overlay_p is None:
                break
            ok_line = (
                "以下点位几何已合格，如无需配合拓扑调整请尽量保持原坐标不要乱改：" + "、".join(ok_points) + "\n"
                if ok_points else ""
            )
            fix_prompt = (
                "我把原失败场景（图1）和你刚才在图2上画出的 3 个起点（星标 S1/S2/S3）一并发给你对照（附图1为原场景，附图2为标注后的图2）。\n"
                "几何检查发现以下问题：\n- " + "\n- ".join(self_check_issues) + "\n"
                + ok_line + "\n"
                "请同时对照图1的来车支路方向以及图2上的 5m/10m 网格，修正有问题的起点坐标：\n"
                f"1. 冲突角色为 {role}。每个 S1/S2/S3 须落在亮青色线，并按该角色核对深青箭头所示的合法行驶方向。\n"
                "2. 每个点独立看图2的 5m/10m 网格读取，严禁等差数列外推，已合格的 S 点请保持原坐标。\n"
                "只输出一个合法 JSON 对象：{\"attack_plans\": [{\"start_xy\": [x, y], \"rationale\": \"...\"}]}"
            )
            raw_fix = client.chat([
                {"role": "system", "content": "只输出一个合法 JSON 对象，不要 Markdown 代码块。"},
                {"role": "user", "content": [
                    {"type": "text", "text": fix_prompt},
                    {"type": "image_url", "image_url": {"url": _data_url(source_bev)}},
                    {"type": "image_url", "image_url": {"url": _data_url(overlay_p)}},
                ]},
            ])
            resp_name = "vlm_self_check_response.txt" if sc_round == 0 else f"vlm_self_check_response_r{sc_round + 1}.txt"
            (run_dir / resp_name).write_text(raw_fix if isinstance(raw_fix, str) else "")
            fixed = parse_json_object(raw_fix)
            if isinstance(fixed, dict) and isinstance(fixed.get("attack_plans"), list) and len(fixed["attack_plans"]) == len(plans):
                plan["attack_plans"] = fixed["attack_plans"]
            else:
                break

    if self_check_issues:
        decision.update(
            suitability="needs_information",
            checks={**decision["checks"], "lanes": "unknown"},
            reasons=self_check_issues,
            adapted_plan=None,
            target_window_s=None,
            retrieval_suggestion=None,
        )

    attack_input = None
    if decision["suitability"] == "ready":
        attack_input = _build_attack_input(data, rendered, decision)

    payload = {
        "suitability": decision["suitability"],
        "reasons": decision["reasons"],
        "checks": decision["checks"],
        "target_window_s": decision["target_window_s"],
        "adapted_plan": decision["adapted_plan"],
        "retrieval_suggestion": decision["retrieval_suggestion"],
        "retrieved_bev_image": rendered["bev_image"],
        "attack_input": attack_input,
        "evidence": {
            "source_bev_image": str(source_bev),
            "retrieved_bev_image": rendered["bev_image"],
            "ego_trajectory_source": ego_traj_source,
            "baseline_frames_path": str(baseline_sim_frames) if baseline_sim_frames else None,
            "retrieved_trajectories": rendered["trajectories_path"],
            "eval_frames": rendered["eval_frames"],
            "ahead_or_side_vehicle_count": len(rendered.get("nearby_vehicles") or []),
            "behind_only_vehicle_count": int(rendered.get("behind_only_count") or 0),
        },
    }
    return payload, getattr(client, "model", "unknown")


def run(input_data: dict) -> dict:
    started = perf_counter()
    llm_client = input_data.get("_llm_client")
    clean = {k: v for k, v in input_data.items() if k != "_llm_client"}
    try:
        data = Input.model_validate(clean)
    except Exception as error:
        return {
            "status": "error", "tool": "edit_planner",
            "error": {"type": "input_validation_error", "message": "edit_planner input failed schema validation",
                      "details": validation_details(error)},
            "meta": {"mode": "real", "model": None, "attempts": 0, "elapsed_seconds": round(perf_counter() - started, 4)},
        }

    run_dir = RECORDS_ROOT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        evaluated, model_name = _evaluate_one(data, run_dir, llm_client=llm_client)
        result = Output.model_validate({
            **evaluated,
            "scene_id": data.scene_ref.scene_id,
            "scene_token": data.scene_ref.scene_token,
            "candidate_id": data.candidate_id,
            "target_scene": data.scenario_result.target_scene,
            "failure_context": data.scenario_result.failure_context.model_dump(mode="json"),
        }).model_dump(mode="json")
    except Exception as error:
        return {
            "status": "error", "tool": "edit_planner",
            "error": {"type": "edit_planning_error", "message": str(error)},
            "meta": {"mode": "real", "model": None, "attempts": 1, "elapsed_seconds": round(perf_counter() - started, 4),
                     "record_dir": str(run_dir)},
        }

    return {
        "status": "success",
        "tool": "edit_planner",
        "result": result,
        "meta": {"mode": "real", "model": model_name, "attempts": 1,
                 "elapsed_seconds": round(perf_counter() - started, 4), "record_dir": str(run_dir)},
    }
