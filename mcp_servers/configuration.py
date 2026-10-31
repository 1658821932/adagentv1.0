"""Non-secret integration settings are owned by the component using them."""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def settings(component):
    path = Path(component) / "adapter/settings.json"
    return json.loads(path.read_text()) if path.is_file() else {}

def runtime_path(component, key, env_name, default):
    value = os.getenv(env_name) or settings(component).get("paths", {}).get(key) or default
    path = Path(value).expanduser()
    # Do not dereference a venv interpreter symlink: doing so launches the
    # system Python and bypasses that environment's site-packages.
    return Path(os.path.abspath(path if path.is_absolute() else PROJECT_ROOT / path))
