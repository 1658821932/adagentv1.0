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



def _resolve_python() -> str:
    """Pick a Python that can run wfa (needs numpy/opencv/Pillow)."""
    python = runtime_path(COMPONENT_ROOT, "python", "WFA_PYTHON",
                          "environments/simengine/.conda/bin/python")
    if not python.is_file():
        raise FileNotFoundError(f"WFA interpreter not found: {python}; set WFA_PYTHON or adapter/settings.json")
    return str(python)



def _check_secret() -> Path:
    secret = Path(os.environ.get("WFA_SECRET_FILE", str(COMPONENT_ROOT / ".secrets" / "vlm.json")))
    if not secret.is_file():
        raise RuntimeError(
            f"VLM secret not found: {secret}. "
            "Create it with keys endpoint/model/api_key and chmod 0600. See README."
        )
    if secret.stat().st_mode & 0o077:
        raise RuntimeError(f"VLM secret must have mode 0600: {secret}")
    return secret
