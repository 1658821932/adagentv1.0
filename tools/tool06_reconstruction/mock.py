"""Mock DGGT reconstruction implementation matching the real Tool interface."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(input_data: dict) -> dict:
    video_path = input_data.get("video_path")
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("reconstruction requires a non-empty 'video_path' string")

    mock_root = (PROJECT_ROOT / "outputs" / "mock" / "reconstruction").resolve()
    point_clouds = mock_root / "point_clouds"
    scene_dir = mock_root / "001"

    return {
        "status": "success",
        "tool": "reconstruction",
        "result": {
            "video_path": video_path.strip(),
            "run_dir": str((PROJECT_ROOT / "logs" / "mock_reconstruction").resolve()),
            "input_dir": str((mock_root / "input").resolve()),
            "output_dir": str(mock_root),
            "scene_ply": str((point_clouds / "combined_001.ply").resolve()),
            "static_ply": str((point_clouds / "static_001.ply").resolve()),
            "dynamic_ply": str((point_clouds / "dynamic_001.ply").resolve()),
            "sky_ply": str((point_clouds / "sky_001.ply").resolve()),
            "camera_params": str((scene_dir / "ply_camera_params.json").resolve()),
            "render_camera_params": str((scene_dir / "camera_params.json").resolve()),
            "supersplat_camera_poses": str((scene_dir / "supersplat_camera_poses.json").resolve()),
            "rendered_video": str((scene_dir / "rendered_video.mp4").resolve()),
            "mode": "mock",
        },
    }
