"""Configuration for the integrated video retrieval capability."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = ("default", "waymo")


@dataclass
class SearchConfig:
    platform: str = "bilibili"
    local_first: bool = True
    match_threshold: float = 0.41

    per_query: int = 3
    max_results: int = 3
    precheck_limit: int = 3
    download_limit: int = 1
    sample_clips: int = 1
    frames: int = 4
    min_confidence: float = 0.75
    scene_threshold: float = 0.12
    chunk_duration: int = 12
    search_results: int = 5
    cookies_browser: str = ""

    # Legacy fields retained for API compatibility; adapters use runtime task paths.
    input_dir: Path = field(default_factory=lambda: (COMPONENT_ROOT / "input").resolve())
    library_dir: Path = field(default_factory=lambda: Path(os.getenv("SENTRY_LIBRARY_DIR", str(COMPONENT_ROOT / "data"))).expanduser().resolve())
    output_dir: Path = field(default_factory=lambda: (COMPONENT_ROOT / "output").resolve())
    logs_dir: Path = field(default_factory=lambda: (COMPONENT_ROOT / "logs").resolve())
    runtime_home: Path = field(default_factory=lambda: Path(os.getenv("SENTRY_INDEX_HOME", str(COMPONENT_ROOT / "runtime"))).expanduser().resolve())
    preprocess_index_chunks: bool = False
    import_source_videos: bool = True

    # OSS rerank staging settings (non-sensitive; credentials stay in .env).
    oss_bucket: str = "ghc-spjs"
    oss_endpoint: str = "https://oss-cn-shanghai.aliyuncs.com"
    oss_prefix: str = "adagent-rerank/"

    @classmethod
    def for_library(cls, library: str) -> SearchConfig:
        if library == "default":
            return cls()
        if library == "waymo":
            root = Path(os.getenv("SENTRY_WAYMO_ROOT", str(COMPONENT_ROOT / "data_waymo_front"))).expanduser()
            return cls(
                chunk_duration=10,
                runtime_home=(root / "runtime").resolve(),
                preprocess_index_chunks=True,
                import_source_videos=False,
            )
        raise ValueError(f"Unknown video library: {library}")

    def validate(self) -> None:
        if self.platform not in {"bilibili", "youtube", "both"}:
            raise ValueError(f"Unsupported platform: {self.platform}")
        if not 0 <= self.match_threshold <= 1:
            raise ValueError("match_threshold must be between 0 and 1")
        for name in (
            "per_query", "max_results", "precheck_limit", "download_limit",
            "sample_clips", "frames", "chunk_duration", "search_results",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 0 <= self.scene_threshold <= 1:
            raise ValueError("scene_threshold must be between 0 and 1")
