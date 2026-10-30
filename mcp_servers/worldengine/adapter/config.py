"""Configuration for the external WorldEngine project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
AGENT_PYTHON = PROJECT_ROOT / "environments" / "agent" / ".venv" / "bin" / "python"
SIMENGINE_PYTHON = PROJECT_ROOT / "environments" / "simengine" / ".conda" / "bin" / "python"
ALGENGINE_PYTHON = PROJECT_ROOT / "environments" / "algengine" / ".conda" / "bin" / "python"
AGENT_FRAME_COUNT = 40


@dataclass
class WorldEngineConfig:
    project_root: Path = Path(os.environ.get("WORLDENGINE_ROOT", str(Path(__file__).resolve().parents[1] / "core"))).expanduser().resolve()
    script: Path | None = None
    timeout: int = int(os.environ.get("WORLDENGINE_TIMEOUT", "86400"))
    # Legacy fields retained for API compatibility; adapters use runtime task paths.
    output_dir: Path = PROJECT_ROOT / "mcp_servers" / "worldengine" / "outputs"
    logs_dir: Path = PROJECT_ROOT / "mcp_servers" / "worldengine" / "logs"

    def testing_script(self) -> Path:
        return self.script or Path(__file__).resolve().with_name("run_testing.sh")
