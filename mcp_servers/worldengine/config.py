"""Configuration for the external WorldEngine project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class WorldEngineConfig:
    project_root: Path = Path(os.environ.get("WORLDENGINE_ROOT", str(Path(__file__).resolve().parent / "core"))).expanduser().resolve()
    script: Path | None = None
    timeout: int = int(os.environ.get("WORLDENGINE_TIMEOUT", "86400"))
    output_dir: Path = PROJECT_ROOT / "mcp_servers" / "worldengine" / "outputs"
    logs_dir: Path = PROJECT_ROOT / "mcp_servers" / "worldengine" / "logs"

    def testing_script(self) -> Path:
        return self.script or self.project_root / "projects" / "SimEngine" / "scripts" / "run_testing.sh"
