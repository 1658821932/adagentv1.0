#!/usr/bin/env python3
"""Build one bound MP4 per scene/camera and resume indexing into the default library."""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import random
import requests
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from api_config import get_profile
from mcp_servers.runtime import write_json
from mcp_servers.sentry_video_search.core.api_settings import bind_profiles
from mcp_servers.sentry_video_search.core.config import SearchConfig
from mcp_servers.sentry_video_search.core.vendor.sentrysearch.chunker import _get_ffmpeg_executable, _get_video_duration
from mcp_servers.sentry_video_search.core.vendor.sentrysearch.qwen_api_embedder import QwenAPIEmbedder
from mcp_servers.sentry_video_search.core.vendor.sentrysearch.store import SentryStore
from mcp_servers.sentry_video_search.adapter.scene_binding import register_scene_video


class Gate:
    def __init__(self, rpm):
        self.lock, self.next_time, self.interval = threading.Lock(), 0., 60. / rpm

    def wait(self):
        with self.lock:
            delay = self.next_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_time = time.monotonic() + self.interval


def camera_frames(sample, camera):
    manifest = json.loads((sample / "manifest.json").read_text())
    frames = json.loads((sample / "metadata/frames.json").read_text())
    if not frames:
        raise ValueError("Scene contains no frames")
    times = [int(f["timestamp"]) for f in frames]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Scene timestamps must increase")
    images = []
    for frame in frames:
        image = Path(frame["cams"][camera]["data_path"])
        images.append(image if image.is_absolute() else sample / "original_images" / image)
    if len(images) < 2:
        raise ValueError("At least two source frames are required")
    return manifest, times, images


def encode_camera(sample, camera, output_root, ffmpeg):
    manifest, times, images = camera_frames(sample, camera)
    token = manifest.get("anchor_token") or manifest.get("scene_token")
    if not token or token != sample.name:
        raise ValueError("Sample directory and anchor token disagree")
    signature = hashlib.sha256(json.dumps({
        "encoding": "h264-original-size-crf23-vfr-ms-v1", "times": times,
        "images": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in images]
    }, sort_keys=True).encode()).hexdigest()
    output = output_root / token / (camera + ".mp4")
    sidecar = output.with_suffix(".scene.json")
    if output.is_file() and sidecar.is_file():
        old = json.loads(sidecar.read_text())
        if old.get("source_signature") == signature and output.stat().st_size > 0:
            write_json(sample / "preview" / ("indexed_" + camera + ".scene.json"), old)
            return output, float(old["duration_seconds"]), signature, True
    output.parent.mkdir(parents=True, exist_ok=True)
    # concat demuxer image framerate sets millisecond precision for real timestamps.
    with tempfile.TemporaryDirectory(prefix="camera-", dir=output.parent) as temp:
        listing = Path(temp) / "frames.ffconcat"
        lines = ["ffconcat version 1.0"]
        durations = [(b-a)/1_000_000 for a,b in zip(times,times[1:])]
        durations.append(durations[-1])
        for image, duration in zip(images, durations):
            escaped = str(image).replace("'", "'\\''")
            lines += ["file '" + escaped + "'", "option framerate 1000", f"duration {duration:.6f}"]
        listing.write_text("\n".join(lines) + "\n")
        temporary = Path(temp) / "camera.mp4"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-threads", "1",
            "-f", "concat", "-safe", "0", "-i", str(listing), "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "2", "-fps_mode", "vfr",
            "-video_track_timescale", "1000000", "-movflags", "+faststart", str(temporary)]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        # Fully decode and check the frame count, rather than trusting an existing filename.
        check = subprocess.run([ffmpeg,"-v","error","-threads","1","-i",str(temporary),
            "-map","0:v:0","-progress","pipe:1","-f","null","-"],
            capture_output=True,text=True,timeout=180,check=True)
        counts = [int(x.split("=",1)[1]) for x in check.stdout.splitlines() if x.startswith("frame=")]
        if not counts or counts[-1] != len(images):
            raise ValueError(f"Decoded frames {counts[-1:]}, expected {len(images)}")
        duration = _get_video_duration(str(temporary))
        expected = (times[-1]-times[0])/1_000_000
        if abs(duration-expected) > 1.0:
            raise ValueError(f"Video duration {duration} does not match source span {expected}")
        temporary.replace(output)
    register_scene_video(output, sample, camera)
    record = json.loads(sidecar.read_text())
    record.update(source_signature=signature, frame_count=len(images), timestamps_us=times,
        source_span_seconds=(times[-1]-times[0])/1_000_000, duration_seconds=duration,
        timestamp_reference="relative to first source frame; MP4 timestamps quantized to 1ms",
        image_size="original", codec="h264", pixel_format="yuv420p")
    write_json(sidecar, record)
    write_json(sample / "preview" / ("indexed_" + camera + ".scene.json"), record)
    return output, duration, signature, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/navtrain_ep_per1")
    parser.add_argument("--cameras", nargs="+", default=["CAM_F0"], help="Camera names to index (default: front only)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rpm", type=int, default=45)
    parser.add_argument("--retries", type=int, default=4, help="Retries for transient embedding network failures")
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 0 or not 1 <= args.rpm <= 50:
        parser.error("workers must be positive; rpm must be 1..50")
    load_dotenv(ROOT / ".env", override=False)
    dataset = args.dataset.expanduser().resolve()
    config = SearchConfig.for_library("default")
    output_root = config.library_dir / (dataset.name + "_multicam")
    progress_dir = dataset / "video_index"
    progress_dir.mkdir(parents=True, exist_ok=True)
    lock = (progress_dir / "batch.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A camera-video indexing job is already active")
    tasks = []
    scenes = sorted(p for p in (dataset / "samples").iterdir() if p.is_dir())
    # Test the already verified scene first, then deterministic scene/camera ordering.
    scenes.sort(key=lambda p: (p.name != "1b25a65f83935c6d", p.name))
    if args.limit_scenes:
        scenes = scenes[:args.limit_scenes]
    for sample in scenes:
        cameras = json.loads((sample / "metadata/cameras.json").read_text())
        missing = set(args.cameras) - set(cameras)
        if missing:
            raise ValueError(f"Scene {sample.name} missing requested cameras: {sorted(missing)}")
        tasks.extend((sample, camera) for camera in args.cameras)
    ffmpeg = _get_ffmpeg_executable()
    profile = get_profile("retrieval", require_key=not args.encode_only)
    store = SentryStore(db_path=config.runtime_home / ".sentrysearch/db",backend="qwen_api")
    initial = store.collection.get(include=["metadatas"])
    existing = {m["source_file"]:m for m in initial["metadatas"]}
    gate = Gate(args.rpm)
    embedder = QwenAPIEmbedder() if not args.encode_only else None
    counters = {"total_scenes":len(scenes),"total_videos":len(tasks),"processed":0,
        "encoded_new":0,"reused_videos":0,"indexed_new":0,"already_indexed":0,"failed":0}
    started = time.time()
    errors = []
    def status(state, **extra):
        write_json(progress_dir / "progress.json",{**counters,"status":state,"pid":os.getpid(),
            "dataset":str(dataset),"video_root":str(output_root),
            "index_path":str(config.runtime_home / ".sentrysearch/db"),
            "embedding_model":profile["embedding_model"], "embedding_dimensions":profile["embedding_dimensions"],
            "started_at":datetime.fromtimestamp(started,timezone.utc).isoformat(),
            "updated_at":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":round(time.time()-started,1),**extra})
    def work(task):
        sample,camera=task
        video,duration,signature,reused=encode_camera(sample,camera,output_root,ffmpeg)
        row={"scene_token":sample.name,"camera":camera,"video_path":str(video),
             "duration_seconds":duration,"source_signature":signature,"reused":reused}
        if args.encode_only:return row,None
        old=existing.get(str(video))
        if old and old.get("source_signature")==signature and old.get("embedding_model")==profile["embedding_model"]:
            return row,None
        for attempt in range(args.retries + 1):
            try:
                gate.wait()
                vector=embedder.embed_video_chunk(str(video))
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt >= args.retries:
                    raise
                delay = min(120., 10. * 2 ** attempt) + random.uniform(0., 3.)
                print(json.dumps({"retry_scene":sample.name,"camera":camera,
                    "attempt":attempt+1,"delay_seconds":round(delay,1),
                    "error_type":type(exc).__name__}), flush=True)
                time.sleep(delay)
        if len(vector)!=profile["embedding_dimensions"]:
            raise ValueError("Embedding dimension mismatch")
        return row,vector
    write_json(progress_dir / "errors.json", [])
    status("running")
    print(json.dumps({"started":True,**counters,"video_root":str(output_root)},ensure_ascii=False),flush=True)
    with (progress_dir / "events.jsonl").open("a", buffering=1) as events:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(work,task):task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                sample,camera=futures[future]
                try:
                    row,vector=future.result()
                    counters["reused_videos" if row["reused"] else "encoded_new"]+=1
                    if vector is not None:
                        video=row["video_path"]
                        store.add_chunk(store.make_chunk_id(video,0.0),vector,
                            {"source_file":video,"start_time":0.,"end_time":row["duration_seconds"],
                             "scene_token":sample.name,"camera":camera,"source_signature":row["source_signature"],
                             "embedding_model":profile["embedding_model"],"sample_root":str(sample)})
                        counters["indexed_new"]+=1
                    elif not args.encode_only:counters["already_indexed"]+=1
                    events.write(json.dumps({"status":"ok",**row,"indexed":not args.encode_only},ensure_ascii=False)+"\n")
                except Exception as exc:
                    counters["failed"]+=1
                    message=str(exc)
                    for key in [profile.get("api_key")]:
                        if key:message=message.replace(key,"[REDACTED]")
                    error={"scene_token":sample.name,"camera":camera,"type":type(exc).__name__,"message":message[:1500]}
                    errors.append(error);write_json(progress_dir/"errors.json",errors)
                    events.write(json.dumps({"status":"error",**error},ensure_ascii=False)+"\n")
                    lower = message.lower()
                    if any(x in lower for x in ["status=401", "status=403", "invalidapikey", "insufficientbalance", "quotaexceeded", "arrearage"]) or len(errors) >= 20:
                        for pending in futures:
                            pending.cancel()
                        status("blocked", error="Indexing stopped after authentication/quota or repeated failures")
                        raise RuntimeError("Batch stopped; inspect errors.json and resume after fixing the cause")
                counters["processed"]+=1
                status("running",last_scene=sample.name,last_camera=camera)
                if counters["processed"]%10==0:print(json.dumps(counters),flush=True)
    status("completed" if not errors else "completed_with_errors")
    print(json.dumps(counters),flush=True)
    return 1 if errors else 0

if __name__=="__main__":
    with bind_profiles(get_profile):
        raise SystemExit(main())
