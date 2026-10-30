"""Configuration for the external DGGT reconstruction capability."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
AGENT_PYTHON = PROJECT_ROOT / "environments" / "agent" / ".venv" / "bin" / "python"
AGENT_FRAME_COUNT = 40


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


def _default_dggt_root() -> Path:
    return _path_from_env("DGGT_ROOT", COMPONENT_ROOT / "core")


@dataclass
class ReconstructionConfig:
    dggt_root: Path = field(default_factory=_default_dggt_root)
    dggt_python: Path = field(
        default_factory=lambda: _path_from_env(
            "DGGT_PYTHON",
            PROJECT_ROOT / "environments" / "dggt" / ".conda" / "bin" / "python",
        )
    )
    segformer_python: Path = field(
        default_factory=lambda: _path_from_env(
            "DGGT_SEGFORMER_PYTHON",
            PROJECT_ROOT / "environments" / "dggt_segformer" / ".conda" / "bin" / "python",
        )
    )
    checkpoint: Path = field(
        default_factory=lambda: _path_from_env(
            "DGGT_CHECKPOINT",
            _default_dggt_root() / "pretrained" / "model_latest_waymo.pth",
        )
    )

    # Legacy fields retained for API compatibility; adapters use runtime task paths.
    input_dir: Path = field(default_factory=lambda: (COMPONENT_ROOT / "input").resolve())
    output_dir: Path = field(default_factory=lambda: (COMPONENT_ROOT / "output").resolve())
    logs_dir: Path = field(default_factory=lambda: (COMPONENT_ROOT / "logs").resolve())

    scene_name: str = "000"
    num_frames: int = 40
    output_fps: float = 8.0
    device: str = "cuda:0"

    def validate(self) -> None:
        required = {
            "DGGT root": self.dggt_root,
            "DGGT Python": self.dggt_python,
            "SegFormer Python": self.segformer_python,
            "DGGT checkpoint": self.checkpoint,
            "SegFormer model": self.dggt_root / "pretrained" / "segformer_hf",
            "extract_video_frames.py": self.dggt_root / "tools" / "extract_video_frames.py",
            "inference.py": self.dggt_root / "inference.py",
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing DGGT runtime paths:\n" + "\n".join(missing))
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.output_fps <= 0:
            raise ValueError("output_fps must be positive")
