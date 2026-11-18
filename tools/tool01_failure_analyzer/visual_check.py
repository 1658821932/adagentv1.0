"""Visual verification of the measured failure facts.

The measured facts are authoritative; this step may not change them. It checks them
against the eight-camera images of one archived scene and fills the environment
fields that the archive has no data for.
"""
from __future__ import annotations

import base64
import json
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
REQUIRED_FIELDS = ("matched", "notes", "weather", "extra")

PROMPT = """核对给定结论并补充环境，不修改结论：
{facts}

首次违规帧为第 {violation} 帧。你会收到 {images_count} 张八相机3×3总览图（中央留白），按时间排序，标有相机名、帧号和相对时间：
{images}
选自首帧、首次违规前后及末帧，重合去重，越界省略。

核对道路形态、车辆位置及相对关系：一致则 matched=true、notes=[]；不一致则 matched=false，notes 逐条写明结论与可见画面的矛盾。
补充 weather（clear/cloudy/rainy/foggy/snowy/unknown）和一句环境描述 extra；无法判断时分别填 unknown、空字符串，不猜测。
必须返回四字段JSON：
{{"matched": true, "notes": [], "weather": "unknown", "extra": ""}}
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
    """First frame, the frames either side of the first violation, and the last frame."""
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
    wanted = (0, violation_step - 1, violation_step + 1, count - 1)
    return sorted({index for index in wanted if 0 <= index < count})


def frame_roles(index, count, violation_step):
    """Which sampling roles this frame plays; roles do coincide at the boundaries."""
    roles = []
    if index == 0:
        roles.append("首帧")
    if index == violation_step - 1:
        roles.append("首次违规前一帧")
    if index == violation_step:
        roles.append("首次违规帧")
    if index == violation_step + 1:
        roles.append("首次违规后一帧")
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


def data_url(path):
    """Encode the JPEG already produced by contact_sheet without recompressing it."""
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


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
        indices = sheet_indices(len(rows), violation)
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

    payload = {key: facts[key] for key in ("failure", "attack") if key in facts}
    listed = "\n".join(
        f"- 第 {position} 张：帧 {index}，{frame_time(rows, index):.1f}s"
        f"（{frame_roles(index, len(rows), violation)}）"
        for position, index in enumerate(indices, start=1))
    content = [{"type": "text", "text": PROMPT.format(
        facts=json.dumps(payload, ensure_ascii=False), violation=violation,
        images_count=len(indices), images=listed)}]
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
        if not isinstance(answer["matched"], bool):
            raise VisualCheckError(f"visual check 'matched' must be a boolean, got {answer['matched']!r}",
                                   stage="vlm")
        notes = answer["notes"]
        if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
            raise VisualCheckError(f"visual check 'notes' must be a list of strings, got {notes!r}",
                                   stage="vlm")
        notes = [note.strip() for note in notes]
        if any(not note for note in notes):
            raise VisualCheckError("visual check notes must be non-empty text", stage="vlm")
        if answer["matched"] and notes:
            raise VisualCheckError("visual check notes must be empty when matched is true", stage="vlm")
        if not answer["matched"] and not notes:
            raise VisualCheckError("visual check must explain each disagreement when matched is false",
                                   stage="vlm")
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
    except (TypeError, ValueError) as error:
        failed("vlm", "invalid model answer")
        raise VisualCheckError(f"visual check returned no usable JSON: {raw[:200]!r}",
                               stage="vlm") from error
    except VisualCheckError as error:
        failed(error.stage, error)
        raise

    result = dict(matched=answer["matched"], notes=notes, weather=weather, extra=extra.strip())
    evidence.write_manifest(run_dir, stage="verification",
                            verification={"matched": result["matched"], "notes": result["notes"]})
    if owned:
        evidence.write_manifest(run_dir, status="success", stage="done", finished_at=evidence.stamp())
    return result
