"""Configuration for the external WorldEngine project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from mcp_servers.configuration import runtime_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
AGENT_PYTHON = runtime_path(COMPONENT_ROOT, "agent_python", "AGENT_PYTHON", "environments/agent/.venv/bin/python")
SIMENGINE_PYTHON = runtime_path(COMPONENT_ROOT, "simengine_python", "SIMENGINE_PYTHON", "environments/simengine/.conda/bin/python")
ALGENGINE_PYTHON = runtime_path(COMPONENT_ROOT, "algengine_python", "ALGENGINE_PYTHON", "environments/algengine/.conda/bin/python")
AGENT_FRAME_COUNT = 40


@dataclass
class WorldEngineConfig:
    project_root: Path = field(default_factory=lambda: Path(os.environ.get("WORLDENGINE_ROOT") or COMPONENT_ROOT / "core").expanduser().resolve())
    script: Path | None = None
    timeout: int = int(os.environ.get("WORLDENGINE_TIMEOUT", "86400"))
    # Legacy fields retained for API compatibility; adapters use runtime task paths.
    output_dir: Path = PROJECT_ROOT / "mcp_servers" / "worldengine" / "outputs"
    logs_dir: Path = PROJECT_ROOT / "mcp_servers" / "worldengine" / "logs"

    def testing_script(self) -> Path:
        return self.script or Path(__file__).resolve().with_name("run_testing.sh")
