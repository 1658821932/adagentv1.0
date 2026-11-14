"""Failure-analysis integration paths and environment resolution."""
import os
from pathlib import Path
from mcp_servers.configuration import runtime_path

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = COMPONENT_ROOT.parents[1]
WFA_ROOT = Path(os.getenv("WFA_ROOT") or COMPONENT_ROOT / "core").expanduser().resolve()
DEFAULT_CONFIG = WFA_ROOT / "configs" / "complete.json"
DEFAULT_CONFIG_EXAMPLE = WFA_ROOT / "configs" / "complete.json.example"
DEFAULT_RUN_NAME = "adagent_failure_analysis"


def runtime_env(env):
    """OpenCV requires host GL libraries even for CPU video decoding."""
    library_dir = runtime_path(COMPONENT_ROOT, "shared_library_dir", "WFA_LIBRARY_DIR",
                               "/usr/lib/x86_64-linux-gnu")
    result = dict(env)
    if library_dir.is_dir():
        entries = [str(library_dir), *result.get("LD_LIBRARY_PATH", "").split(os.pathsep)]
        result["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(p for p in entries if p))
    return result



def _resolve_python() -> str:
    """Pick a Python that can run wfa (needs numpy/opencv/Pillow)."""
    python = runtime_path(COMPONENT_ROOT, "python", "WFA_PYTHON",
                          "environments/simengine/bin/python")
    if not python.is_file():
        raise FileNotFoundError(f"WFA interpreter not found: {python}; set WFA_PYTHON or adapter/settings.json")
    return str(python)
