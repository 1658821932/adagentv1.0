"""Stage 5: second filter for downloaded/local videos."""

from __future__ import annotations

import shutil
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..first_person_detector import classify_video, iter_videos
from .scene_segmenter import source_group_name


@dataclass
class PostFilterConfig:
    library_dir: Path = Path("data/internet_videos")
    frames: int = 4
    max_height: int = 640
    min_confidence: float = 0.75
    import_local: bool = True


def is_generated_scene_video(video: Path) -> bool:
    return "scenes" in video.parts


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def import_source_video(video: Path, *, library_dir: Path) -> tuple[Path, bool]:
    video = video.resolve()
    library_dir = library_dir.expanduser()
    if is_under(video, library_dir):
        return video, False

    target = library_dir / source_group_name(video) / video.name
    target.parent.mkdir(parents=True, exist_ok=True)
    imported = not (target.exists() and target.stat().st_size > 0)
    if imported:
        shutil.copy2(video, target)
    for sidecar in (video.with_suffix(".scene.json"), Path(str(video) + ".scene.json")):
        if not sidecar.is_file():
            continue
        record = json.loads(sidecar.read_text())
        if record.get("sample_root"):
            root = Path(record["sample_root"]).expanduser()
            record["sample_root"] = str((sidecar.parent / root).resolve() if not root.is_absolute() else root.resolve())
        record["video_path"] = str(target.resolve())
        target.with_suffix(".scene.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        break
    return target.resolve(), imported


def source_videos(path: Path) -> list[Path]:
    return [video for video in iter_videos(path.expanduser()) if not is_generated_scene_video(video)]


def filter_one(video: Path, config: PostFilterConfig) -> tuple[Path, dict[str, Any]]:
    input_video = video.resolve()
    imported = False
    if config.import_local:
        video, imported = import_source_video(video, library_dir=config.library_dir)

    result = classify_video(video, frame_count=config.frames, max_height=config.max_height)
    result.accepted = result.is_first_person and result.is_clean_view and result.confidence >= config.min_confidence
    row = asdict(result)
    row.update(
        {
            "input_video": str(input_video),
            "imported_source": imported,
            "accepted": result.accepted,
        }
    )
    return video, row


def filter_sources(path: Path, config: PostFilterConfig) -> list[tuple[Path, dict[str, Any]]]:
    videos = source_videos(path)
    if not videos:
        raise SystemExit(f"No videos found: {path}")
    return [filter_one(video, config) for video in videos]
