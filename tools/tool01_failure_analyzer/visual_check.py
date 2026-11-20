"""Optional environment description from the violation-frame camera overview.

All structural and interaction fields are measured deterministically before
this check. This is the only visual-model step in tools 01-04.
"""
from __future__ import annotations

import base64
from pathlib import Path

from tools.llm_runtime import parse_json_object

CAMERAS = ("CAM_L0", "CAM_F0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_B0", "CAM_R2")
# Surround layout: front top-centre, back bottom-centre, left column left, right column right.
# The centre cell stays blank, exactly like the archive overview board.
LAYOUT = (
    ("CAM_L0", "CAM_F0", "CAM_R0"),
    ("CAM_L1", None, "CAM_R1"),
    ("CAM_L2", "CAM_B0", "CAM_R2"),
)
TILE_W, TILE_H, LABEL_H = 320, 180, 18
IMAGE_QUALITY = 82
WEATHER = ("clear", "cloudy", "rainy", "foggy", "snowy", "unknown")
REQUIRED_FIELDS = ("weather", "extra", "notes")
ENVIRONMENT_FORBIDDEN = ("直路", "弯道", "丁字", "十字", "路口", "汇流", "车道",
                         "自车", "他车", "车辆", "左转", "右转", "直行")


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


class VisualCheckError(RuntimeError):
    """The images could not be prepared or the model returned an unusable answer."""

    def __init__(self, message, *, stage="images", timeout=False):
        super().__init__(message)
        self.stage = stage
        self.timeout = timeout


def camera_images(facts):
    """Frame-ordered image paths per camera, supplied by the fact extractor."""
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
    """Two frames before, the violation, and two frames after; clamp at boundaries."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise VisualCheckError(f"scene frame count must be a positive integer, got {count!r}",
                               stage="input")
    if isinstance(violation_step, bool) or not isinstance(violation_step, int):
        raise VisualCheckError(f"violation frame must be an integer, got {violation_step!r}",
                               stage="input")
    if not 0 <= violation_step < count:
        raise VisualCheckError(
            f"violation frame {violation_step} is outside the recorded frames 0..{count - 1}",
            stage="input")
    wanted = (violation_step - 2, violation_step, violation_step + 2)
    return sorted({max(0, min(count - 1, index)) for index in wanted})


def frame_roles(index, count, violation_step):
    """Which sampling roles this frame plays; roles do coincide at the boundaries."""
    roles = []
    if index == 0:
        roles.append("首帧")
    if index == violation_step - 2:
        roles.append("首次违规前二帧")
    if index == violation_step:
        roles.append("首次违规帧")
    if index == violation_step + 2:
        roles.append("首次违规后二帧")
    if index == count - 1:
        roles.append("末帧")
    return "、".join(roles)


def contact_sheet(images, index, target, rows):
    """Compose the eight cameras of one frame into a labelled 3x3 sheet."""
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


def data_url(path, mime="image/jpeg"):
    """Encode an already prepared image without recompressing it."""
    return "data:" + mime + ";base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def check(archive, token, facts, run_dir=None):
    """Verify the measured facts against the images and return the environment fields."""
    from agent.llm_client import LLMClient
    from tools.tool01_failure_analyzer import evidence

    owned = run_dir is None
    if owned:
        run_dir = evidence.new_run_dir(facts.get("simulation_record") or archive, token)
    run_dir = Path(run_dir)
    evidence.write_manifest(run_dir, tool="failure_analyzer", status="running", stage="images",
                            token=str(token),
                            simulation=facts.get("simulation_record") or str(archive))

    def failed(stage, error):
        evidence.write_manifest(run_dir, status="failed", stage=stage, error=str(error))

    try:
        images, rows = camera_images(facts)
        violation = facts["violation_step"]
        sheet_indices(len(rows), violation)
        indices = [violation]
        evidence.write_manifest(run_dir, violation_step=violation, selected=[
            {"index": index, "time_s": frame_time(rows, index),
             "sources": {camera: str(images[camera][index]) for camera in CAMERAS}}
            for index in indices])
        sheet_dir = run_dir / "sheets"
        sheets = [contact_sheet(images, index, sheet_dir / f"step_{index:02d}.jpg", rows)
                  for index in indices]
        evidence.write_manifest(run_dir, frames=[
            {"index": index, "sheet": str(sheet), "sheet_sha256": evidence.sha256_file(sheet)}
            for index, sheet in zip(indices, sheets)])
        urls = [data_url(sheet) for sheet in sheets]
        evidence.write_manifest(run_dir, sent_images=[
            {"index": index, "sha256": evidence.sent_image_sha256(url)}
            for index, url in zip(indices, urls)])
    except VisualCheckError as error:
        failed(error.stage, error)
        raise
    except (OSError, ValueError) as error:
        failed("images", type(error).__name__)
        raise VisualCheckError(f"image preparation failed ({type(error).__name__})",
                               stage="images") from None

    listed = "\n".join(
        f"- 第 {position} 张：帧 {index}，{frame_time(rows, index):.1f}s"
        f"（{frame_roles(index, len(rows), violation)}）"
        for position, index in enumerate(indices, start=1))
    content = [{"type": "text", "text": PROMPT.format(
        violation=violation, images_count=len(indices), images=listed)}]
    content += [{"type": "image_url", "image_url": {"url": url}} for url in urls]

    evidence.write_manifest(run_dir, stage="vlm")
    try:
        client = LLMClient(component="failure_analyzer")
        raw = client.chat([
            {"role": "system", "content": "只输出一个 JSON 对象，不要 Markdown，不要解释。"},
            {"role": "user", "content": content},
        ])
    except Exception as error:  # never echo request details or credentials
        stage = "vlm_empty" if "empty response" in str(error) else "vlm"
        timed_out = isinstance(error, TimeoutError) or type(error).__name__ in {
            "APITimeoutError", "Timeout", "ReadTimeout", "ConnectTimeout"}
        failed(stage, type(error).__name__)
        raise VisualCheckError(f"model request failed ({type(error).__name__})",
                               stage=stage, timeout=timed_out) from None

    if not isinstance(raw, str) or not raw.strip():
        failed("vlm_empty", "model returned no text")
        raise VisualCheckError("model returned no text", stage="vlm_empty")

    response_file = run_dir / "response.txt"
    response_file.write_text(raw)
    evidence.write_manifest(run_dir, stage="verification", response_file=str(response_file))

    try:
        answer = parse_json_object(raw)
        if not isinstance(answer, dict):
            raise VisualCheckError(f"visual check returned no JSON object: {raw[:200]!r}", stage="vlm")
        missing = [name for name in REQUIRED_FIELDS if name not in answer]
        if missing:
            raise VisualCheckError(f"visual check response is missing {', '.join(missing)}: {raw[:200]!r}",
                                   stage="vlm")
        extra_fields = sorted(set(answer) - set(REQUIRED_FIELDS))
        if extra_fields:
            raise VisualCheckError(
                f"visual check response has unexpected fields {', '.join(extra_fields)}: {raw[:200]!r}",
                stage="vlm")
        notes = answer["notes"]
        if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
            raise VisualCheckError(f"visual check 'notes' must be a list of strings, got {notes!r}",
                                   stage="vlm")
        notes = [note.strip() for note in notes]
        if any(not note for note in notes):
            raise VisualCheckError("visual check notes must be non-empty text", stage="vlm")
        report = answer["weather"]
        if not isinstance(report, str):
            raise VisualCheckError(f"visual check 'weather' must be a string, got {report!r}", stage="vlm")
        weather = report.strip().lower()
        if weather not in WEATHER:
            raise VisualCheckError(f"visual check 'weather' must be one of {', '.join(WEATHER)}, got {report!r}",
                                   stage="vlm")
        extra = answer["extra"]
        if not isinstance(extra, str):
            raise VisualCheckError(f"visual check 'extra' must be a string, got {extra!r}", stage="vlm")
        forbidden = environment_forbidden_terms(extra)
        if forbidden:
            raise VisualCheckError(
                f"visual check 'extra' contains non-environment terms: {', '.join(forbidden)}",
                stage="vlm")
        for note in notes:
            forbidden = environment_forbidden_terms(note)
            if forbidden:
                raise VisualCheckError(
                    f"visual check notes contain non-environment terms: {', '.join(forbidden)}",
                    stage="vlm")
    except (TypeError, ValueError) as error:
        failed("vlm", "invalid model answer")
        raise VisualCheckError(f"visual check returned no usable JSON: {raw[:200]!r}",
                               stage="vlm") from error
    except VisualCheckError as error:
        failed(error.stage, error)
        raise

    result = {
        "sample_frames": indices,
        "corrections": [],
        "notes": notes,
        "weather": weather,
        "extra": extra.strip(),
    }
    evidence.write_manifest(run_dir, stage="verification",
                            verification={key: result[key] for key in
                                          ("sample_frames", "corrections", "notes")})
    if owned:
        evidence.write_manifest(run_dir, status="success", stage="done", finished_at=evidence.stamp())
    return result
