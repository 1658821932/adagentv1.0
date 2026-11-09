"""Configuration for the external WorldEngine project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from mcp_servers.configuration import runtime_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
AGENT_PYTHON = runtime_path(COMPONENT_ROOT, "agent_python", "AGENT_PYTHON", "environments/agent/.venv/bin/python")
SIMENGINE_PYTHON = runtime_path(COMPONENT_ROOT, "simengine_python", "SIMENGINE_PYTHON", "environments/simengine/bin/python")
ALGENGINE_PYTHON = runtime_path(COMPONENT_ROOT, "algengine_python", "ALGENGINE_PYTHON", "environments/algengine/bin/python")
SHARED_LIBRARY_DIR = runtime_path(COMPONENT_ROOT, "shared_library_dir", "WORLDENGINE_LIBRARY_DIR", "/usr/lib/x86_64-linux-gnu")
FFMPEG = runtime_path(COMPONENT_ROOT, "ffmpeg", "WORLDENGINE_FFMPEG", "/usr/bin/ffmpeg")
REPLAY_RENDER_PYTHON = runtime_path(COMPONENT_ROOT, "replay_render_python", "WORLDENGINE_RENDER_PYTHON", "environments/scene_render/bin/python")
NUPLAN_SOURCE = runtime_path(COMPONENT_ROOT, "nuplan_source", "WORLDENGINE_NUPLAN_SOURCE", "environments/simengine/lib/python3.9/site-packages/nuplan")
GSPLAT_EXTENSION = runtime_path(COMPONENT_ROOT, "gsplat_extension", "WORLDENGINE_GSPLAT_EXTENSION", "mcp_servers/scene_editing/runtime/case1_reliability_fix01/prebuilt-gsplat/gsplat_cuda.so")
AGENT_FRAME_COUNT = 40


@dataclass
class WorldEngineConfig:
    project_root: Path = field(default_factory=lambda: Path(os.environ.get("WORLDENGINE_ROOT") or COMPONENT_ROOT / "core").expanduser().resolve())
    script: Path | None = None
    timeout: int = int(os.environ.get("WORLDENGINE_TIMEOUT", "86400"))

    def testing_script(self) -> Path:
        return self.script or Path(__file__).resolve().with_name("run_testing.sh")
