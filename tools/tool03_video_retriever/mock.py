"""Mock video retrieval implementation with the same outer schema as the real Tool."""

from __future__ import annotations

from pathlib import Path

from .schemas import Input


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(input_data: dict) -> dict:
    data = Input.model_validate(input_data)
    path = Path(data.mock_video_path) if data.mock_video_path else PROJECT_ROOT / "outputs/mock/video.mp4"
    video_path = str(path.expanduser().resolve())

    return {
        "status": "success",
        "tool": "video_retriever",
        "result": {
            "requirement": data.requirement,
            "video_path": video_path,
            "run_dir": str((PROJECT_ROOT / "logs" / "mock_video_retriever").resolve()),
            "mode": "mock",
            "source_file": video_path,
            "scene_ref": None,
            "candidate_id": data.candidate_id,
            "match_score": 1.0,
            "timings": {"total_seconds": 0.0},
        },
    }
