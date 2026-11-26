"""Visual checks for Tool 01: BEV scenario classification and 8-camera environment."""
from __future__ import annotations

import base64
import shutil
from pathlib import Path

from tools.llm_runtime import parse_json_object

CAMERAS = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_B0", "CAM_R2")
LAYOUT = (
    ("CAM_L0", "CAM_F0", "CAM_R0"),
    ("CAM_L1", None, "CAM_R1"),
    ("CAM_L2", "CAM_B0", "CAM_R2"),
)
TILE_W, TILE_H, LABEL_H = 320, 180, 18
IMAGE_QUALITY = 82
WEATHER = ("clear", "cloudy", "rainy", "foggy", "snowy", "unknown")
MAP_CONTEXTS = ("straight_road", "curve", "t_junction", "cross_intersection", "merge", "roundabout")
FAILURE_DRIVERS = ("collision_with_vehicle", "collision_non_vehicle", "departure_by_other", "departure_ego_only")
BEV_REQUIRED_FIELDS = ("map_context", "failure_driver", "ego_behavior", "other_behavior", "summary")
REQUIRED_FIELDS = ("weather", "extra", "notes")
ENVIRONMENT_FORBIDDEN = ("直路", "弯道", "丁字", "十字", "路口", "汇流", "车道",
                         "自车", "他车", "车辆", "左转", "右转", "直行")
NEARBY_PROMPT_LIMIT_M = 10.0
COLLISION_LOCK_M = 0.5


def environment_forbidden_terms(value):
    return [term for term in ENVIRONMENT_FORBIDDEN if term in value]


PROMPT = """你的唯一任务是根据首次违规帧的八相机总览补充天气和环境描述。
不要判断道路形态、路口类型、车辆、车头方向、车道保持、他车来向、车辆动作或交互类型。
首次违规帧为第 {violation} 帧。你会收到 {images_count} 张八相机总览图：
{images}
只填写：
- weather：clear / cloudy / rainy / foggy / snowy / unknown
- extra：一句可见环境描述，只写天气、光照、能见度、路面状态和静态背景；没有可靠描述时为空字符串
- notes：可选补充说明；没有补充时为空数组

看不清天气时 weather 填 unknown。不要返回任何其他字段。
必须返回 JSON：
{{"weather":"unknown","extra":"","notes":[]}}
"""


BEV_PROMPT_TEMPLATE = """你是自动驾驶闭环失败场景分析员。请严格根据这张 BEV 轨迹图和下方实测物理摘要，输出客观的场景定性与简短剧本。

【图例】
- 橙色车身与轨迹：自车（EGO），标有帧号 E0..En；亮红粗框为首次违规帧。
- 蓝色车身与轨迹：周边他车；黑色箭头为车头方向。
- 浅灰填充：官方车道；黄色细线：车道中心线；黑色粗边界：官方路口区域。

【实测物理摘要（程序硬测量，必须严格服从，不可推翻）】
- 官方首次违规类型：{outcome}
- 首次违规帧：第 {violation_step} 帧（t = {violation_time_s}s）
- 自车运动（0..{violation_step}帧）：{ego_summary}
- 违规帧近距他车：
{nearby_summary}

【输出 JSON 格式（严格只含以下 5 个字段）】
{{
  "map_context": "straight_road | curve | t_junction | cross_intersection | merge | roundabout",
  "failure_driver": "collision_with_vehicle | collision_non_vehicle | departure_by_other | departure_ego_only",
  "ego_behavior": "≤40字：客观描述自车全程到违规帧的运动变化、车道偏移与所处道路位置",
  "other_behavior": "≤60字：仅描述导致本次失败的唯一目标车的方位、运动变化与相对时机；无目标车时填 \\"\\"",
  "summary": "≤100字：1句话串联 最小道路形态 + 自车行为 + 唯一目标车行为或自身偏离 + 第{violation_step}帧结果"
}}

【通用原则】
1. 最小道路形态：map_context 填写复刻亮红粗框处失败所需的最小道路几何，忽略违规前经过或远处与本次失败无关的道路结构。
2. 致因归属：
   - collision → collision_with_vehicle（与他车碰撞，含静止车）或 collision_non_vehicle（无他车接触）；
   - road_departure → departure_by_other（受某辆他车逼迫/占道）或 departure_ego_only（自车自身问题）。
3. 唯一目标车：other_behavior 和 summary 只描述导致本次失败的唯一一辆目标车，不提无关背景车；无目标车时 other_behavior 为 ""。
4. 客观忠实：禁止编造图与摘要之外的信息（如红绿灯、天气、光线、主观评价），所有速度、静止与左右转向方向严格以摘要为准。"""


class VisualCheckError(RuntimeError):
    """Images could not be prepared or the model returned an invalid answer."""

    def __init__(self, message, *, stage="images", timeout=False):
        super().__init__(message)
        self.stage = stage
        self.timeout = timeout


def _yaw_text(deg):
    """Explicit direction wording so VLM never misreads math signs."""
    val = abs(float(deg))
    if val < 5.0:
        return f"近直行（偏转{val:.1f}°）"
    return f"左转{val:.1f}°" if deg > 0 else f"右转{val:.1f}°"


def _rel_heading_text(deg):
    if deg is None:
        return "未知方向"
    a = abs(float(deg))
    if a <= 45.0:
        return "同向"
    if a >= 135.0:
        return "对向"
    return "横向"


def select_prompt_vehicles(facts):
    """Select the exact nearby vehicles passed to BEV VLM (no silent fallback)."""
    outcome = facts["outcome"]
    candidates = []
    for v in facts.get("nearby_vehicles") or []:
        vd = v.get("violation_distance_m")
        if vd is None or vd > NEARBY_PROMPT_LIMIT_M or not v.get("kinematics"):
            continue
        candidates.append((float(vd), v))
    candidates.sort(key=lambda item: (item[0], str(item[1].get("track") or "")))
    if outcome == "collision" and candidates and candidates[0][0] <= COLLISION_LOCK_M:
        return [candidates[0][1]]
    return [v for _, v in candidates[:3]]


def build_bev_prompt(facts, selected_vehicles):
    viol = int(facts["violation_step"])
    rows = facts.get("frame_images") or []
    t_s = float(rows[viol]["time_s"]) if 0 <= viol < len(rows) else round(viol * 0.5, 1)
    ego_k = facts["ego_kinematics"]
    ego_summary = (
        f"速度 {ego_k['speed_start_mps']:.1f}→{ego_k['speed_violation_mps']:.1f} m/s，"
        f"位移 {ego_k['displacement_m']:.1f} m，"
        f"全程{_yaw_text(ego_k['yaw_change_deg'])}，"
        f"末段1秒{_yaw_text(ego_k['tail_yaw_deg'])}"
    )
    lines = []
    for idx, v in enumerate(selected_vehicles, start=1):
        vd = float(v["violation_distance_m"])
        vk = v["kinematics"]
        pos = v.get("bearing_at_violation") or "附近"
        size = v.get("size_m") or [4.5, 2.0, 1.6]
        vtype = f"大型车（长{size[0]:.1f}m）" if size[0] >= 6.5 else f"普通车（长{size[0]:.1f}m）"
        if vk["is_stationary"]:
            motion = "静止"
        else:
            motion = (
                f"速度{vk['speed_start_mps']:.1f}→{vk['speed_violation_mps']:.1f}m/s，"
                f"全程{_yaw_text(vk['yaw_change_deg'])}，"
                f"末段1秒{_yaw_text(vk['tail_yaw_deg'])}"
            )
        contact = "（接触）" if vd <= 0.05 else ""
        rel_txt = _rel_heading_text(v.get("relative_heading_at_violation_deg"))
        lines.append(f"  {idx}. {pos} {vd:.1f}m{contact}：{rel_txt}，{motion}，{vtype}")
    nearby_summary = "\n".join(lines) if lines else "  无（10m 内无其他车辆）"
    return BEV_PROMPT_TEMPLATE.format(
        outcome=facts["outcome"],
        violation_step=viol,
        violation_time_s=f"{t_s:.1f}",
        ego_summary=ego_summary,
        nearby_summary=nearby_summary,
    )


def data_url(path, mime="image/jpeg"):
    return "data:" + mime + ";base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def check_bev(facts, run_dir):
    """Classify road context, failure driver, and script from BEV + physical summary."""
    from agent.llm_client import LLMClient
    from tools.tool01_failure_analyzer import evidence

    source = Path(str(facts.get("bev_image") or ""))
    if not source.is_file():
        raise VisualCheckError("facts do not contain a readable BEV evidence image", stage="bev")
    try:
        return _check_bev(facts, source, Path(run_dir), shutil.copyfile, LLMClient, evidence)
    finally:
        if source.exists():
            source.unlink()


def _check_bev(facts, source, run_dir, copyfile, client_type, evidence):
    target = run_dir / "bev_overview.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    copyfile(source, target)
    selected = select_prompt_vehicles(facts)
    prompt = build_bev_prompt(facts, selected)
    url = data_url(target, "image/png")
    evidence.write_manifest(run_dir, stage="bev_vlm", bev_image=str(target))
    try:
        client = client_type(component="failure_analyzer")
        raw = client.chat([
            {"role": "system", "content": "只输出一个合法 JSON 对象，不要 Markdown 代码块，不要任何多余解释。"},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ]},
        ])
    except Exception as error:
        timed_out = isinstance(error, TimeoutError) or type(error).__name__ in {
            "APITimeoutError", "Timeout", "ReadTimeout", "ConnectTimeout"}
        evidence.write_manifest(run_dir, stage="bev_vlm", error=type(error).__name__)
        raise VisualCheckError("BEV model request failed (" + type(error).__name__ + ")",
                               stage="bev_vlm", timeout=timed_out) from None
    (run_dir / "bev_response.txt").write_text(raw if isinstance(raw, str) else "")
    if not isinstance(raw, str) or not raw.strip():
        evidence.write_manifest(run_dir, stage="bev_vlm", error="empty response")
        raise VisualCheckError("BEV model returned no text", stage="bev_vlm")
    try:
        answer = parse_json_object(raw)
    except (TypeError, ValueError) as error:
        evidence.write_manifest(run_dir, stage="bev_vlm", error="invalid JSON")
        raise VisualCheckError("BEV model returned invalid JSON", stage="bev_vlm") from error
    if not isinstance(answer, dict) or set(answer) != set(BEV_REQUIRED_FIELDS):
        raise VisualCheckError(
            "BEV model response must contain exact keys " + ", ".join(BEV_REQUIRED_FIELDS),
            stage="bev_vlm",
        )
    context = answer["map_context"]
    driver = answer["failure_driver"]
    ego_behavior = answer["ego_behavior"]
    other_behavior = answer["other_behavior"]
    summary = answer["summary"]
    if context not in MAP_CONTEXTS:
        raise VisualCheckError(f"BEV model map_context is invalid: {context!r}", stage="bev_vlm")
    if driver not in FAILURE_DRIVERS:
        raise VisualCheckError(f"BEV model failure_driver is invalid: {driver!r}", stage="bev_vlm")
    if not isinstance(ego_behavior, str) or not ego_behavior.strip():
        raise VisualCheckError("BEV model ego_behavior must be non-empty text", stage="bev_vlm")
    if not isinstance(other_behavior, str):
        raise VisualCheckError("BEV model other_behavior must be a string", stage="bev_vlm")
    if not isinstance(summary, str) or not summary.strip():
        raise VisualCheckError("BEV model summary must be non-empty text", stage="bev_vlm")

    # Strict cross-validation against official PDM outcome and physical candidates.
    outcome = facts["outcome"]
    if outcome == "collision" and not driver.startswith("collision_"):
        raise VisualCheckError(f"official outcome is collision but BEV returned failure_driver={driver!r}", stage="bev_vlm")
    if outcome == "road_departure" and not driver.startswith("departure_"):
        raise VisualCheckError(f"official outcome is road_departure but BEV returned failure_driver={driver!r}", stage="bev_vlm")
    if outcome == "road_departure" and not selected and driver != "departure_ego_only":
        raise VisualCheckError(f"no vehicle within 10m for road_departure but BEV returned {driver!r}", stage="bev_vlm")
    if driver in ("departure_ego_only", "collision_non_vehicle") and other_behavior.strip():
        raise VisualCheckError(f"failure_driver={driver!r} must have empty other_behavior", stage="bev_vlm")
    if driver in ("collision_with_vehicle", "departure_by_other"):
        if not selected:
            raise VisualCheckError(f"failure_driver={driver!r} requires a nearby vehicle within 10m", stage="bev_vlm")
        if not other_behavior.strip():
            raise VisualCheckError(f"failure_driver={driver!r} requires non-empty other_behavior", stage="bev_vlm")

    result = {
        "bev_image": str(target),
        "map_context": context,
        "failure_driver": driver,
        "ego_behavior": ego_behavior.strip(),
        "other_behavior": other_behavior.strip(),
        "summary": summary.strip(),
        "selected_vehicle": selected[0] if driver in ("collision_with_vehicle", "departure_by_other") else None,
    }
    evidence.write_manifest(run_dir, stage="bev_verification", decisions={
        "map_context": context, "failure_driver": driver})
    return result


# ------------------------------------------------------------------ 8-camera env
def camera_images(facts):
    rows = facts.get("frame_images")
    if not rows:
        raise VisualCheckError("facts carry no frame_images mapping")
    images = {camera: [] for camera in CAMERAS}
    for row in rows:
        for camera in CAMERAS:
            path = Path((row.get("images") or {}).get(camera) or "")
            if not path.is_file():
                raise VisualCheckError(f"missing image for {camera} frame {row['index']}: {path}")
            images[camera].append(path)
    return images, rows


def frame_time(rows, index):
    return float(rows[index]["time_s"])


def sheet_indices(count, violation_step):
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise VisualCheckError(f"scene frame count must be positive int, got {count!r}", stage="input")
    if isinstance(violation_step, bool) or not isinstance(violation_step, int) or not 0 <= violation_step < count:
        raise VisualCheckError(f"violation frame {violation_step!r} is outside 0..{count - 1}", stage="input")
    wanted = (violation_step - 2, violation_step, violation_step + 2)
    return sorted({max(0, min(count - 1, idx)) for idx in wanted})


def contact_sheet(images, index, target, rows):
    from PIL import Image, ImageDraw, ImageFont

    sheet = Image.new("RGB", (TILE_W * 3, (TILE_H + LABEL_H) * 3), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for row, cells in enumerate(LAYOUT):
        for column, camera in enumerate(cells):
            if camera is None:
                continue
            left, top = column * TILE_W, row * (TILE_H + LABEL_H)
            with Image.open(images[camera][index]) as tile:
                sheet.paste(tile.convert("RGB").resize((TILE_W, TILE_H)), (left, top))
            draw.text((left + 6, top + TILE_H + 1),
                      f"{camera} frame {index} ({frame_time(rows, index):.1f}s)",
                      fill=(0, 0, 0), font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="JPEG", quality=IMAGE_QUALITY)
    return target


def check(archive, token, facts, run_dir=None):
    """Supplement weather and static environmental context from the violation frame."""
    from agent.llm_client import LLMClient
    from tools.tool01_failure_analyzer import evidence

    owned = run_dir is None
    if owned:
        run_dir = evidence.new_run_dir(facts.get("simulation_record") or archive, token)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence.write_manifest(run_dir, tool="failure_analyzer", status="running", stage="images",
                            token=str(token), simulation=facts.get("simulation_record") or str(archive))

    def failed(stage, error):
        evidence.write_manifest(run_dir, status="failed", stage=stage, error=str(error))

    try:
        images, rows = camera_images(facts)
        violation = facts["violation_step"]
        sheet_indices(len(rows), violation)
        indices = [violation]
        sheet_dir = run_dir / "sheets"
        sheets = [contact_sheet(images, idx, sheet_dir / f"step_{idx:02d}.jpg", rows) for idx in indices]
        urls = [data_url(sheet) for sheet in sheets]
    except VisualCheckError as error:
        failed(error.stage, error)
        raise
    except (OSError, ValueError) as error:
        failed("images", type(error).__name__)
        raise VisualCheckError(f"image preparation failed ({type(error).__name__})", stage="images") from None

    listed = f"- 第 1 张：帧 {violation}，{frame_time(rows, violation):.1f}s（首次违规帧）"
    content = [{"type": "text", "text": PROMPT.format(violation=violation, images_count=1, images=listed)}]
    content += [{"type": "image_url", "image_url": {"url": url}} for url in urls]

    evidence.write_manifest(run_dir, stage="vlm")
    try:
        client = LLMClient(component="failure_analyzer")
        raw = client.chat([
            {"role": "system", "content": "只输出一个 JSON 对象，不要 Markdown，不要解释。"},
            {"role": "user", "content": content},
        ])
    except Exception as error:
        stage = "vlm_empty" if "empty response" in str(error) else "vlm"
        timed_out = isinstance(error, TimeoutError) or type(error).__name__ in {
            "APITimeoutError", "Timeout", "ReadTimeout", "ConnectTimeout"}
        failed(stage, type(error).__name__)
        raise VisualCheckError(f"model request failed ({type(error).__name__})", stage=stage, timeout=timed_out) from None

    if not isinstance(raw, str) or not raw.strip():
        failed("vlm_empty", "model returned no text")
        raise VisualCheckError("model returned no text", stage="vlm_empty")
    (run_dir / "response.txt").write_text(raw)

    try:
        answer = parse_json_object(raw)
        if not isinstance(answer, dict) or set(answer) != set(REQUIRED_FIELDS):
            raise VisualCheckError(f"visual check response keys invalid: {raw[:200]!r}", stage="vlm")
        notes = answer["notes"]
        if not isinstance(notes, list) or not all(isinstance(n, str) and n.strip() for n in notes):
            raise VisualCheckError("visual check notes must be a list of non-empty strings", stage="vlm")
        notes = [n.strip() for n in notes]
        weather = answer["weather"].strip().lower() if isinstance(answer["weather"], str) else ""
        if weather not in WEATHER:
            raise VisualCheckError(f"visual check weather invalid: {answer['weather']!r}", stage="vlm")
        extra = answer["extra"]
        if not isinstance(extra, str):
            raise VisualCheckError("visual check extra must be a string", stage="vlm")
        for text in (extra, *notes):
            forbidden = environment_forbidden_terms(text)
            if forbidden:
                raise VisualCheckError(f"visual check contains non-environment terms: {', '.join(forbidden)}", stage="vlm")
    except (TypeError, ValueError) as error:
        failed("vlm", "invalid model answer")
        raise VisualCheckError(f"visual check returned no usable JSON: {raw[:200]!r}", stage="vlm") from error
    except VisualCheckError as error:
        failed(error.stage, error)
        raise

    result = {"sample_frames": indices, "notes": notes, "weather": weather, "extra": extra.strip()}
    if owned:
        evidence.write_manifest(run_dir, status="success", stage="done", finished_at=evidence.stamp())
    return result
