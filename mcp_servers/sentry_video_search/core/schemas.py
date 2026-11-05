"""Native retrieval result schema."""
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class VideoSearchResult:
    requirement: str
    video_path: str
    run_dir: str
    mode: str
    match_score: float
    timings: dict[str, float]
    source_file: str | None = None
    source_segment: dict[str, float] | None = None
    scene_ref: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
