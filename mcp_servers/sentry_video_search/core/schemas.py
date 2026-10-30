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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
