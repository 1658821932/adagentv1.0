"""Mock video retrieval implementation with the same outer schema as the real Tool."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(input_data: dict) -> dict:
    requirement = input_data.get("requirement")
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("video_retriever requires a non-empty 'requirement' string")

    mock_video_path = input_data.get("mock_video_path")
    if isinstance(mock_video_path, str) and mock_video_path.strip():
        video_path = str(Path(mock_video_path).expanduser().resolve())
    else:
        video_path = str((PROJECT_ROOT / "outputs" / "mock" / "video.mp4").resolve())

    return {
        "status": "success",
        "tool": "video_retriever",
        "result": {
            "requirement": requirement.strip(),
            "video_path": video_path,
            "run_dir": str((PROJECT_ROOT / "logs" / "mock_video_retriever").resolve()),
            "mode": "mock",
            "match_score": 1.0,
            "timings": {"total_seconds": 0.0},
        },
    }
