"""Single-view DGGT Mode 2 reconstruction service."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .config import ReconstructionConfig
from mcp_servers.runtime import runtime_task, current_run


from .schemas import ReconstructionResult

def _log_tail(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:].strip()


def _run_command(
    command: list[str], *, cwd: Path, log_path: Path, python_bin: Path | None = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if python_bin is not None:
        env["PATH"] = str(python_bin.parent) + os.pathsep + env.get("PATH", "")
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        tail = _log_tail(log_path)
        message = f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        if tail:
            message += f"\n\nLast log output:\n{tail}"
        raise RuntimeError(message)


@runtime_task("dggt", "reconstruct")
def reconstruct(
    video_path: str | Path,
    edit_config_path: str | Path | None = None,
    config: ReconstructionConfig | None = None,
    ego_trajectory_offsets_path: str | Path | None = None,
    export_dynamic_frame_ply: bool = False,
    num_frames: int | None = None,
    export_render_state: bool = False,
) -> ReconstructionResult:
    """Prepare one MP4 and reconstruct it with DGGT Mode 2, single view."""

    if export_render_state and edit_config_path is not None:
        raise ValueError("Native render-state export currently requires unedited reconstruction")
    config = config or ReconstructionConfig()
    if num_frames is not None:
        config.num_frames = int(num_frames)
    config.validate()

    source_video = Path(video_path).expanduser().resolve()
    if not source_video.is_file():
        compact = "".join(source_video.name.split())
        for candidate in source_video.parent.iterdir() if source_video.parent.is_dir() else []:
            if "".join(candidate.name.split()) == compact:
                source_video = candidate
                break
    if not source_video.is_file():
        raise FileNotFoundError(f"Input video not found: {source_video}")
    probe = subprocess.run(
        [str(config.segformer_python), "-c",
         "import cv2,json,sys; cap=cv2.VideoCapture(sys.argv[1]); "
         "print(json.dumps({'fps':cap.get(cv2.CAP_PROP_FPS),"
         "'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT))})); cap.release()",
         str(source_video)],
        capture_output=True, text=True, check=True,
    )
    video_info = json.loads(probe.stdout)
    source_fps = float(video_info["fps"])
    if source_fps <= 0 or int(video_info["frames"]) < config.num_frames:
        raise ValueError(f"Video must have a valid FPS and at least {config.num_frames} frames")
    config.output_fps = source_fps
    if ego_trajectory_offsets_path is not None:
        ego_trajectory_offsets = Path(ego_trajectory_offsets_path).expanduser().resolve()
        if not ego_trajectory_offsets.is_file():
            raise FileNotFoundError(f"DGGT ego trajectory offsets not found: {ego_trajectory_offsets}")

    task = current_run("dggt")
    input_root, output_root, run_dir = task.input, task.output, task.root

    preprocess_command = [
        str(config.segformer_python),
        str(config.dggt_root / "tools" / "extract_video_frames.py"),
        str(source_video),
        str(input_root),
        "--scene-name",
        config.scene_name,
        "--num-frames",
        str(config.num_frames),
        "--duration-seconds",
        str(config.num_frames / source_fps),
        "--device",
        config.device,
    ]
    _run_command(
        preprocess_command,
        cwd=config.dggt_root,
        log_path=task.logs / "01_preprocess.log",
        python_bin=config.segformer_python,
    )

    reconstruction_command = [
        str(config.dggt_python),
        str(config.dggt_root / "inference.py"),
        "--image_dir",
        str(input_root),
        "--scene_names",
        config.scene_name,
        "--input_views",
        "1",
        "--sequence_length",
        str(config.num_frames),
        "--start_idx",
        "0",
        "--mode",
        "2",
        "--ckpt_path",
        str(config.checkpoint),
        "--output_path",
        str(output_root),
        "--output_fps",
        str(config.output_fps),
        "-images",
        "-export_ply",
        "--export_camera_params",
    ]
    if edit_config_path is not None:
        edit_config = Path(edit_config_path).expanduser().resolve()
        if not edit_config.is_file():
            raise FileNotFoundError(f"DGGT edit config not found: {edit_config}")
        reconstruction_command.extend(["--edit_config", str(edit_config)])
    if ego_trajectory_offsets_path is not None:
        reconstruction_command.extend(["--ego_trajectory_offsets", str(ego_trajectory_offsets)])
    if export_dynamic_frame_ply:
        reconstruction_command.append("--export_dynamic_frame_ply")
    if export_render_state:
        reconstruction_command.append("--export_render_state")
    _run_command(
        reconstruction_command,
        cwd=config.dggt_root,
        log_path=task.logs / "02_reconstruction.log",
        python_bin=config.dggt_python,
    )

    # DGGT timestamps are model-normalized, not seconds from the source video.
    # Keep them for Gaussian rendering and record real capture times separately.
    for manifest_path in output_root.glob("point_clouds/dynamic_frames_*.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for frame in manifest:
            frame["source_timestamp_seconds"] = int(frame["frame_index"]) / source_fps
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    point_cloud_dir = output_root / "point_clouds"
    combined_files = sorted(point_cloud_dir.glob("combined_*.ply"))
    if len(combined_files) != 1:
        raise RuntimeError(
            f"Expected exactly one combined PLY in {point_cloud_dir}, found {len(combined_files)}"
        )

    scene_ply = combined_files[0]
    output_scene_name = scene_ply.stem.removeprefix("combined_")
    scene_dir = output_root / output_scene_name

    static_ply = point_cloud_dir / f"static_{output_scene_name}.ply"
    dynamic_ply = point_cloud_dir / f"dynamic_{output_scene_name}.ply"
    sky_ply = point_cloud_dir / f"sky_{output_scene_name}.ply"
    camera_params = scene_dir / "ply_camera_params.json"
    render_camera_params = scene_dir / "camera_params.json"
    supersplat_camera_poses = scene_dir / "supersplat_camera_poses.json"
    rendered_video = scene_dir / "rendered_video.mp4"

    required_outputs = [
        scene_ply,
        static_ply,
        sky_ply,
        camera_params,
        render_camera_params,
        supersplat_camera_poses,
        rendered_video,
    ]
    if export_render_state:
        required_outputs.append(scene_dir / "render_state.pt")
    missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(
            "DGGT finished but required reconstruction artifacts are missing:\n"
            + "\n".join(missing_outputs)
        )

    result = ReconstructionResult(
        video_path=str(source_video),
        run_dir=str(run_dir),
        input_dir=str(input_root),
        input_scene_name=config.scene_name,
        output_dir=str(output_root),
        scene_ply=str(scene_ply),
        static_ply=str(static_ply),
        dynamic_ply=str(dynamic_ply) if dynamic_ply.is_file() else None,
        sky_ply=str(sky_ply),
        camera_params=str(camera_params),
        render_camera_params=str(render_camera_params),
        supersplat_camera_poses=str(supersplat_camera_poses),
        rendered_video=str(rendered_video),
        render_state=str(scene_dir / "render_state.pt") if export_render_state else None,
    )

    (output_root / "result.json").write_text(
        json.dumps({"status": "success", **result.to_dict()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


# Stable Agent-facing facade; heavyweight execution still uses its subprocess.
from .client import reconstruct_video
from .instances import discover_instances
from .editing import edit_scene
