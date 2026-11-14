#!/usr/bin/env python3
"""为数据集的 preview/original_front_2hz.mp4 补 scene sidecar。

让 scene_binding 能把 CAM_F0 也绑定到数据集内的视频（而不是检索库的索引原视频）。
幂等：已存在则跳过；纯文本操作，不做视频编码。

运行: /root/miniconda3/envs/dggt_py310/bin/python make_front_sidecar.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEARCH_ROOT = Path(
    "/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagent"
    "/mcp_servers/sentry_video_search"
)
MULTICAM = SEARCH_ROOT / "data/navtrain_ep_per1_multicam"


def process(scene_dir: Path) -> str:
    token = scene_dir.name
    side_f0 = scene_dir / "CAM_F0.scene.json"
    if not side_f0.is_file():
        return f"[skip] {token}: no CAM_F0.scene.json"
    sample = Path(json.loads(side_f0.read_text())["sample_root"])
    video = sample / "preview" / "original_front_2hz.mp4"
    if not video.is_file():
        return f"[skip] {token}: no original_front_2hz.mp4"
    out = sample / "preview" / "original_front_2hz.scene.json"
    if out.is_file():
        return f"[skip] {token}: sidecar exists"

    frames = json.loads((sample / "metadata" / "frames.json").read_text())
    ts = [int(frame["timestamp"]) for frame in frames]
    data_paths = [frame["cams"]["CAM_F0"]["data_path"] for frame in frames]
    signature = hashlib.sha256(
        json.dumps(
            {"scene_token": token, "camera": "CAM_F0",
             "timestamps_us": ts, "data_paths": data_paths},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    payload = {
        "video_path": str(video.resolve()),
        "camera": "CAM_F0",
        "scene_token": token,
        "sample_root": str(sample.resolve()),
        "source_signature": signature,
        "frame_count": len(frames),
        "timestamps_us": ts,
        "source_span_seconds": round((ts[-1] - ts[0]) / 1e6, 6) if len(ts) > 1 else 0.0,
        "duration_seconds": round(len(frames) / 2.0, 6),
        "timestamp_reference": "relative to first source frame; MP4 timestamps quantized to 1ms",
        "image_size": "original",
        "codec": "h264",
        "pixel_format": "yuv420p",
    }
    out.write_text(json.dumps(payload, indent=1))
    return f"[ok] {token}"


def main() -> None:
    scenes = sorted(p for p in MULTICAM.iterdir() if p.is_dir())
    counts = {"ok": 0, "skip": 0}
    for scene_dir in scenes:
        line = process(scene_dir)
        counts["ok" if line.startswith("[ok]") else "skip"] += 1
        if line.startswith("[ok]"):
            print(line, flush=True)
    print("done: ok=%d skip=%d total=%d" % (counts["ok"], counts["skip"], len(scenes)), flush=True)


if __name__ == "__main__":
    main()
