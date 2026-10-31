"""Validate edited-scene artifacts and require an explicit visual evaluator."""

from __future__ import annotations

import json
import os
import base64
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from .config import DEFAULT_BASE_URL, DEFAULT_MODEL, SETTINGS, frame_python, vlm_settings
from .schemas import VisualEvaluation
from pydantic import ValidationError
from tools.llm_runtime import validation_details


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Qwen-VL response must be a JSON object")
    return value


def _sample_frames(video: Path, count: int) -> list[Path]:
    directory = Path(tempfile.mkdtemp(prefix="scene_validator_frames_"))
    ffprobe = "ffprobe"
    ffmpeg = "ffmpeg"
    if not (subprocess.run(["which", ffprobe], capture_output=True).returncode == 0 and subprocess.run(["which", ffmpeg], capture_output=True).returncode == 0):
        frame_interpreter = frame_python()
        script = (
            "import cv2,sys\n"
            "video,out,count=sys.argv[1],sys.argv[2],int(sys.argv[3])\n"
            "cap=cv2.VideoCapture(video)\n"
            "total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
            "if total <= 0: raise RuntimeError('video has no frames')\n"
            "for i in range(max(1,count)):\n"
            " index=round((total-1)*i/max(count-1,1))\n"
            " cap.set(cv2.CAP_PROP_POS_FRAMES,index)\n"
            " ok,frame=cap.read()\n"
            " if not ok: raise RuntimeError('failed to read video frame')\n"
            " cv2.imwrite(f'{out}/frame_{i:03d}.jpg',frame)\n"
            "cap.release()\n"
        )
        subprocess.run([frame_interpreter, "-c", script, str(video), str(directory), str(count)], check=True)
        return [directory / f"frame_{i:03d}.jpg" for i in range(max(1, count))]
    probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video)], capture_output=True, text=True, check=True)
    duration = max(float(probe.stdout.strip()), 0.0)
    if duration <= 0:
        raise RuntimeError(f"could not determine video duration: {video}")
    frames: list[Path] = []
    for idx in range(max(1, count)):
        frame = directory / f"frame_{idx:03d}.jpg"
        timestamp = duration * idx / max(count - 1, 1)
        if idx == count - 1:
            timestamp = max(0.0, duration - min(0.125, duration / max(count * 2, 1)))
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale='min(640,iw)':-2", "-q:v", "3", str(frame)], check=True)
        frames.append(frame)
    return frames


def _image_data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _dashscope_chat(base_url: str, api_key: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if body.get("error"):
        raise RuntimeError(f"DashScope API error: {body['error']}")
    return body


def _edited_value(input_data: dict[str, Any], key: str) -> Any:
    edited = input_data.get("edited_scene") or input_data.get("scene") or {}
    if isinstance(edited, dict) and key in edited:
        return edited[key]
    return input_data.get(key)


def _evaluate_locally(input_data: dict[str, Any], video: Path, requirement: Any) -> dict[str, Any] | None:
    supplied = input_data.get("visual_evaluation")
    if isinstance(supplied, dict):
        return supplied
    command = input_data.get("local_evaluator") or os.environ.get("SCENE_VALIDATOR_LOCAL_COMMAND")
    if not command:
        return None
    proc = subprocess.run(command, shell=True, input=json.dumps({"rendered_video": str(video), "scene_requirement": requirement}, ensure_ascii=False), text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "local scene evaluator failed")
    result = json.loads(proc.stdout)
    if not isinstance(result, dict):
        raise ValueError("local scene evaluator must return a JSON object")
    return result


def _evaluate_qwen(input_data: dict[str, Any], video: Path, requirement: Any) -> dict[str, Any]:
    settings = vlm_settings()
    supplied_frames = input_data.get("frame_paths")
    if isinstance(supplied_frames, list) and supplied_frames:
        frames = [Path(item).expanduser().resolve() for item in supplied_frames]
        missing = [str(item) for item in frames if not item.is_file()]
        if missing:
            raise FileNotFoundError("scene_validator frame_paths missing: " + ", ".join(missing))
        cleanup = False
    else:
        # Eight uniformly spaced frames give the visual evaluator coverage of
        # the full edit while keeping the multimodal request manageable.
        frames = _sample_frames(video, int(input_data.get("frame_count", 8)))
        cleanup = True
    try:
        prompt = (
            "只评估编辑是否满足最初场景需求：目标车辆是否清晰可见、位置和尺度是否合理、是否接地、"
            "跨帧是否连续，以及目标附近是否有编辑引入的伪影。以下是已知的 DGGT 基础重建缺陷，"
            "不属于本次编辑质量，不要因此判定失败，也不要放入 issues："
            + "；".join(input_data.get("ignored_reconstruction_artifacts", [])) + "。"
            "可将它们记入 ignored_reconstruction_artifacts。若缺陷遮住目标，导致无法确认编辑效果，仍应判定失败。"
            "逐张检查提供的关键帧，证据注明帧序号；对于前车切入，确认切入后不继续向右越过自车车道，"
            "并检查末帧车辆仍可辨认；只看到车辆接近画面中心不足以证明它在车道内。"
            "若任一关键帧车辆明显悬空、出道路或破碎，"
            "geometry_valid 或 visual_valid 必须为 false，并写入 issues。"
            "不要从画面推断配置文件中的数值尺度、精确车距、自车已刹车或闭环策略决策；"
            "这些只能从相应轨迹和仿真结果验证。严格只输出 JSON："
            '{"requirement_consistency":true,"visual_valid":true,"geometry_valid":true,'
            '"physics_valid":true,"traffic_logic_valid":true,"confidence":0.0,'
            '"evidence":[],"issues":[],"ignored_reconstruction_artifacts":[]}。\n'
            "最初场景需求：" + json.dumps(requirement, ensure_ascii=False)
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": _image_data_url(frame)}} for frame in frames)
        body = _dashscope_chat(settings.base_url, settings.api_key, {"model": settings.model, "messages": [{"role": "user", "content": content}], "max_tokens": 8192}, settings.timeout)
        result = _extract_json(body["choices"][0]["message"]["content"])
        if not isinstance(result, dict):
            raise ValueError("Qwen-VL scene evaluation must return a JSON object")
        result["model"] = settings.model
        result["sampled_frames"] = len(frames)
        return result
    finally:
        if cleanup:
            parent = frames[0].parent
            for frame in frames:
                try:
                    frame.unlink()
                except OSError:
                    pass
            try:
                parent.rmdir()
            except OSError:
                pass


def run(input_data: dict[str, Any]) -> dict[str, Any]:
    rendered_video = _edited_value(input_data, "rendered_video")
    requirement = input_data.get("scene_requirement") or input_data.get("requirement")
    if not isinstance(rendered_video, str) or not rendered_video.strip():
        raise ValueError("scene_validator requires rendered_video from the previous edit step")
    if not isinstance(requirement, (dict, str)) or not requirement:
        raise ValueError("scene_validator requires the original scene_requirement")
    video = Path(rendered_video).expanduser().resolve()
    issues: list[str] = []
    if not video.is_file():
        issues.append(f"rendered video does not exist: {video}")
    config_path = _edited_value(input_data, "scene_config")
    scene_path = _edited_value(input_data, "scene_ply")
    if config_path:
        path = Path(config_path).expanduser()
        if not path.is_file():
            issues.append(f"scene config does not exist: {path}")
        else:
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                issues.append(f"scene config is not valid JSON: {exc}")
    if scene_path and not Path(scene_path).expanduser().is_file():
        issues.append(f"scene PLY does not exist: {scene_path}")
    visual = _evaluate_locally(input_data, video, requirement) if video.is_file() else None
    if visual is None and video.is_file() and not input_data.get("visual_evaluation") and not (input_data.get("local_evaluator") or os.environ.get("SCENE_VALIDATOR_LOCAL_COMMAND")):
        visual = _evaluate_qwen(input_data, video, requirement)
    if visual is not None:
        try:
            visual = VisualEvaluation.model_validate(visual).model_dump()
        except ValidationError as exc:
            return {"status": "error", "tool": "scene_validator", "error": {
                "type": "output_validation_error",
                "message": "Visual evaluator output must contain five strict boolean decisions",
                "validation_errors": validation_details(exc)}}
    if visual is None:
        issues.append("no visual evaluator configured; scene requirement was not evaluated")
    elif isinstance(visual.get("issues"), list):
        issues.extend(str(item) for item in visual["issues"])
    requirement_consistency = visual["requirement_consistency"] if visual is not None else False
    visual_valid = visual["visual_valid"] if visual is not None else False
    geometry_valid = visual["geometry_valid"] if visual is not None else False
    physics_valid = visual["physics_valid"] if visual is not None else False
    traffic_logic_valid = visual["traffic_logic_valid"] if visual is not None else False
    passed = all((requirement_consistency, visual_valid, geometry_valid, physics_valid, traffic_logic_valid)) and not issues
    return {"status": "success", "tool": "scene_validator", "result": {
        "passed": passed,
        "requirement_consistency": requirement_consistency,
        "geometry_valid": geometry_valid,
        "physics_valid": physics_valid,
        "traffic_logic_valid": traffic_logic_valid,
        "visual_valid": visual_valid,
        "confidence": visual.get("confidence", 0.0) if visual else 0.0,
        "evidence": visual.get("evidence", []) if visual else [],
        "ignored_reconstruction_artifacts": visual.get("ignored_reconstruction_artifacts", []) if visual else [],
        "model": visual.get("model") if visual else None,
        "sampled_frames": visual.get("sampled_frames", 0) if visual else 0,
        "issues": issues,
        "rendered_video": str(video),
    }}
