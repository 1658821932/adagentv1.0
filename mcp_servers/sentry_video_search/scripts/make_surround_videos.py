#!/usr/bin/env python3
"""批量为数据集每个场景生成 7 路侧/后视视频 + 八宫格视频。

落点: data/navtrain_ep_per1/samples/<token>/preview/  (与 original_front_2hz.mp4 同目录)
  original_CAM_<X>_2hz.mp4          7 路单视角, 1920x1080 h264 yuv420p, 无标签(同 original_front)
  original_CAM_<X>_2hz.scene.json   sidecar (scene_binding.py 扫 preview/ 自动绑定进 scene_ref)
  original_eightcam_2hz.mp4         八宫格 3x3, 1920x1080, 每格 640x360 带相机标签

输入: mcp_servers/sentry_video_search/data/navtrain_ep_per1_multicam/<token>/CAM_F0.scene.json
      -> sample_root -> metadata/frames.json (frame["cams"][cam]["data_path"] 逐帧对齐)

幂等: 产物齐全则跳过。多进程并行。
运行: /root/miniconda3/envs/dggt_py310/bin/python make_surround_videos.py [workers]
"""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

FFMPEG = "/root/miniconda3/envs/dggt/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
SEARCH_ROOT = Path("/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/adagent/mcp_servers/sentry_video_search")
MULTICAM = SEARCH_ROOT / "data/navtrain_ep_per1_multicam"

GRID_LAYOUT = [["CAM_L1", "CAM_F0", "CAM_R1"],
               ["CAM_L0", None,     "CAM_R0"],
               ["CAM_L2", "CAM_B0", "CAM_R2"]]
SIDE_CAMS = ["CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0"]
CELL_W, CELL_H = 640, 360
FULL_W, FULL_H = 1920, 1080
FPS = 2.0
GRID_NAME = "original_eightcam_2hz.mp4"


def open_writer(path: Path, w: int, h: int) -> subprocess.Popen:
    return subprocess.Popen(
        [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{FPS:.6f}", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", "-preset", "veryfast", str(path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def label(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (150, 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def ego_cell() -> np.ndarray:
    img = np.zeros((CELL_H, CELL_W, 3), np.uint8)
    cv2.putText(img, "EGO", (CELL_W // 2 - 70, CELL_H // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (90, 90, 90), 3)
    return img


def missing_cell(name: str) -> np.ndarray:
    img = np.full((CELL_H, CELL_W, 3), 40, np.uint8)
    cv2.putText(img, name, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)
    cv2.putText(img, "MISSING", (CELL_W // 2 - 90, CELL_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2)
    return img


def process_scene(scene_dir: Path) -> str:
    token = scene_dir.name
    side_f0 = scene_dir / "CAM_F0.scene.json"
    if not side_f0.is_file():
        return f"[fail] {token}: no CAM_F0.scene.json"
    sample = Path(json.loads(side_f0.read_text())["sample_root"])
    preview = sample / "preview"

    todo_cams = [c for c in SIDE_CAMS
                 if not (preview / f"original_{c}_2hz.mp4").is_file()
                 or not (preview / f"original_{c}_2hz.scene.json").is_file()]
    need_grid = not (preview / GRID_NAME).is_file()
    if not todo_cams and not need_grid:
        return f"[skip] {token}"

    frames = json.loads((sample / "metadata/frames.json").read_text())
    image_root = sample / "original_images"
    ts = [int(f["timestamp"]) for f in frames]

    writers, tmp_paths = {}, {}
    try:
        for cam in todo_cams:
            tmp = preview / f".original_{cam}_2hz.tmp.mp4"
            writers[cam] = (open_writer(tmp, FULL_W, FULL_H), tmp)
            tmp_paths[cam] = tmp
        grid_tmp = None
        if need_grid:
            grid_tmp = preview / f".{GRID_NAME}.tmp.mp4"
            writers["__grid__"] = (open_writer(grid_tmp, CELL_W * 3, CELL_H * 3), grid_tmp)

        for frame in frames:
            imgs = {}
            for cam, info in frame["cams"].items():
                p = Path(info["data_path"])
                p = p if p.is_absolute() else image_root / p
                imgs[cam] = cv2.imread(str(p)) if p.is_file() else None
            for cam in todo_cams:
                img = imgs.get(cam)
                if img is None:
                    img = np.zeros((FULL_H, FULL_W, 3), np.uint8)
                elif img.shape[:2] != (FULL_H, FULL_W):
                    img = cv2.resize(img, (FULL_W, FULL_H))
                writers[cam][0].stdin.write(img.tobytes())
            if "__grid__" in writers:
                rows = []
                for row in GRID_LAYOUT:
                    cells = []
                    for cam in row:
                        if cam is None:
                            cells.append(ego_cell())
                        else:
                            img = imgs.get(cam)
                            if img is None:
                                cells.append(missing_cell(cam))
                            else:
                                img = cv2.resize(img, (CELL_W, CELL_H))
                                label(img, cam)
                                cells.append(img)
                    rows.append(np.hstack(cells))
                writers["__grid__"][0].stdin.write(np.vstack(rows).tobytes())

        ok = True
        for proc, _ in writers.values():
            proc.stdin.close()
            if proc.wait() != 0:
                ok = False
        if not ok:
            for _, tmp in writers.values():
                tmp.unlink(missing_ok=True)
            return f"[fail] {token}: encoder error"

        duration = len(frames) / FPS
        for cam in todo_cams:
            mp4 = preview / f"original_{cam}_2hz.mp4"
            tmp_paths[cam].rename(mp4)
            data_paths = [f["cams"][cam]["data_path"] for f in frames]
            sig = hashlib.sha256(json.dumps(
                {"scene_token": token, "camera": cam,
                 "timestamps_us": ts, "data_paths": data_paths},
                sort_keys=True).encode()).hexdigest()
            (preview / f"original_{cam}_2hz.scene.json").write_text(json.dumps({
                "video_path": str(mp4.resolve()),
                "camera": cam,
                "scene_token": token,
                "sample_root": str(sample.resolve()),
                "source_signature": sig,
                "frame_count": len(frames),
                "timestamps_us": ts,
                "source_span_seconds": round((ts[-1] - ts[0]) / 1e6, 6) if len(ts) > 1 else 0.0,
                "duration_seconds": round(duration, 6),
                "timestamp_reference": "relative to first source frame; MP4 timestamps quantized to 1ms",
                "image_size": "original",
                "codec": "h264",
                "pixel_format": "yuv420p",
            }, indent=1))
        if grid_tmp is not None:
            grid_tmp.rename(preview / GRID_NAME)
        return f"[ok] {token}"
    except Exception as e:  # noqa: BLE001
        for proc, tmp in writers.values():
            try:
                proc.kill()
            except Exception:
                pass
            tmp.unlink(missing_ok=True)
        return f"[fail] {token}: {e}"


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    scenes = sorted(p for p in MULTICAM.iterdir() if p.is_dir())
    print(f"scenes: {len(scenes)}, workers: {workers}", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_scene, s): s for s in scenes}
        for fut in as_completed(futs):
            print(f"{done:4d}/{len(scenes)} {fut.result()}", flush=True)
            done += 1
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
