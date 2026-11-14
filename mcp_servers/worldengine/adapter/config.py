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
# The policy under test is one fixed artifact; keep the default in one place so
# simulation and post-training can never disagree about it.
DEFAULT_POLICY_CONFIG = runtime_path(COMPONENT_ROOT, "policy_config", "WORLDENGINE_POLICY_CONFIG", "mcp_servers/worldengine/core/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py")
DEFAULT_POLICY_CHECKPOINT = runtime_path(COMPONENT_ROOT, "policy_checkpoint", "WORLDENGINE_POLICY_CHECKPOINT", "mcp_servers/worldengine/core/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth")
DEFAULT_MODEL_NAME = "expert_vadv2_50pct"
DEFAULT_DATA_TYPE = "nuplan_openscene"


def runtime_env(environ):
    """Pass configured host GL libraries before native Python processes start."""
    env = dict(environ)
    library = Path(env.get('WORLDENGINE_LIBRARY_DIR') or SHARED_LIBRARY_DIR).expanduser()
    if library.is_dir():
        library = str(library.resolve())
        env['WORLDENGINE_LIBRARY_DIR'] = library
        entries = [library, *env.get('LD_LIBRARY_PATH', '').split(os.pathsep)]
        env['LD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys(p for p in entries if p))
    return env


def resolve_policy_inputs(input_data: dict) -> dict:
    """Fill the fixed policy identity so callers only supply scene-specific data."""
    resolved = dict(input_data)
    resolved.setdefault("config_path", None)
    resolved.setdefault("checkpoint_path", None)
    resolved.setdefault("model_name", None)
    resolved.setdefault("data_type", None)
    if not resolved["config_path"]:
        resolved["config_path"] = str(DEFAULT_POLICY_CONFIG)
    if not resolved["checkpoint_path"]:
        resolved["checkpoint_path"] = str(DEFAULT_POLICY_CHECKPOINT)
    if not resolved["model_name"]:
        resolved["model_name"] = DEFAULT_MODEL_NAME
    if not resolved["data_type"]:
        resolved["data_type"] = DEFAULT_DATA_TYPE
    missing = [name for name in ("config_path", "checkpoint_path")
               if not Path(resolved[name]).expanduser().is_file()]
    if missing:
        raise FileNotFoundError("Policy model artifact is missing: " + ", ".join(
            str(resolved[name]) for name in missing))
    return resolved


@dataclass
class WorldEngineConfig:
    project_root: Path = field(default_factory=lambda: Path(os.environ.get("WORLDENGINE_ROOT") or COMPONENT_ROOT / "core").expanduser().resolve())
    script: Path | None = None
    timeout: int = int(os.environ.get("WORLDENGINE_TIMEOUT", "86400"))

    def testing_script(self) -> Path:
        return self.script or Path(__file__).resolve().with_name("run_testing.sh")
