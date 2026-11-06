"""Fast first-person video classifier using the Kimi visual API.

The module samples a few frames from a local video and asks a multimodal
visual model for a strict JSON classification. It is intended as a fast
pre-index filter, not as a replacement for full video retrieval.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .pipeline.common import chat_completion, model_settings, load_prompt




VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass
class FirstPersonResult:
    video: str
    is_first_person: bool
    is_clean_view: bool
    accepted: bool
    confidence: float
    reason: str
    sampled_frames: int


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"Command failed: {' '.join(cmd)}")
    return proc.stdout.strip()


def get_duration(video_path: Path) -> float:
    output = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    return max(float(output), 0.0)


def sample_frames(video_path: Path, frame_count: int, max_height: int) -> list[Path]:
    duration = get_duration(video_path)
    if duration <= 0:
        raise RuntimeError(f"Could not determine video duration: {video_path}")

    frame_count = max(1, frame_count)
    # Avoid exact endpoints because many videos have less stable first/last frames.
    timestamps = [duration * (i + 1) / (frame_count + 1) for i in range(frame_count)]

    temp_dir = Path(tempfile.mkdtemp(prefix="first_person_frames_"))
    frame_paths: list[Path] = []
    for idx, timestamp in enumerate(timestamps):
        frame_path = temp_dir / f"frame_{idx:03d}.jpg"
        _run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                f"scale='min({max_height},iw)':-2",
                "-q:v",
                "3",
                str(frame_path),
            ]
        )
        if frame_path.exists() and frame_path.stat().st_size > 0:
            frame_paths.append(frame_path)

    if not frame_paths:
        raise RuntimeError(f"No frames sampled from video: {video_path}")
    return frame_paths


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def classify_frames(
    frame_paths: Iterable[Path],
    *,
    video_path: Path,
    model: str | None,
    base_url: str | None,
) -> FirstPersonResult:
    model, base_url, api_key = model_settings(model, base_url)

    content: list[dict] = []
    frame_count = 0
    for frame_path in frame_paths:
        content.append({"type": "image_url", "image_url": {"url": _data_url(frame_path)}})
        frame_count += 1

    prompt = load_prompt("first_person.md")
    content.insert(0, {"type": "text", "text": "以下为待审核视频的关键帧，按顺序排列。"})

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": content}],
        "temperature": 0,
    }
    body = chat_completion(
        base_url=base_url,
        api_key=api_key,
        payload=payload,
    )

    text = body["choices"][0]["message"]["content"]
    parsed = _extract_json(text)
    is_first_person = bool(parsed["is_first_person"])
    is_clean_view = bool(parsed.get("is_clean_view", True))
    confidence = float(parsed.get("confidence", 0.0))
    return FirstPersonResult(
        video=str(video_path),
        is_first_person=is_first_person,
        is_clean_view=is_clean_view,
        accepted=is_first_person and is_clean_view,
        confidence=confidence,
        reason=str(parsed.get("reason", "")),
        sampled_frames=frame_count,
    )


def classify_video(
    video_path: str | Path,
    *,
    frame_count: int = 4,
    max_height: int = 640,
    model: str | None = None,
    base_url: str | None = None,
) -> FirstPersonResult:
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")

    frames = sample_frames(video, frame_count=frame_count, max_height=max_height)
    try:
        return classify_frames(
            frames,
            video_path=video,
            model=model,
            base_url=base_url,
        )
    finally:
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


def iter_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    videos = [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Quickly classify whether videos are first-person.")
    parser.add_argument("path", help="Video file or directory.")
    parser.add_argument("--frames", type=int, default=4, help="Number of frames to sample per video.")
    parser.add_argument("--max-height", type=int, default=640, help="Max sampled frame height.")
    parser.add_argument("--model", default=None, help="Visual model, default kimi-k3.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible model base URL.")
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--min-confidence", type=float, default=0.75, help="Minimum confidence for accepted=true.")
    args = parser.parse_args()

    videos = iter_videos(Path(args.path).expanduser())
    if not videos:
        raise SystemExit(f"No videos found: {args.path}")

    output_path = Path(args.output).expanduser() if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = output_path.open("w", encoding="utf-8") if output_path else None
    try:
        for video in videos:
            result = classify_video(
                video,
                frame_count=args.frames,
                max_height=args.max_height,
                model=args.model,
                base_url=args.base_url,
            )
            result.accepted = result.accepted and result.confidence >= args.min_confidence
            row = asdict(result)
            line = json.dumps(row, ensure_ascii=False)
            print(line)
            if output_file:
                output_file.write(line + "\n")
                output_file.flush()
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
