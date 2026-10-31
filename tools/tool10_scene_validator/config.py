"""Non-secret visual validator defaults; API keys remain in .env."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = json.loads(Path(__file__).with_name("config.json").read_text())
DEFAULT_BASE_URL = SETTINGS["base_url"]
DEFAULT_MODEL = SETTINGS["model"]

def frame_python():
    value = os.getenv("SCENE_VALIDATOR_FRAME_PYTHON") or SETTINGS["frame_python"]
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else ROOT / path)
