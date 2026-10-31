"""Non-secret visual validator defaults; API keys remain in .env."""
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = json.loads(Path(__file__).with_name("config.json").read_text())
DEFAULT_BASE_URL = SETTINGS["base_url"]
DEFAULT_MODEL = SETTINGS["model"]

def frame_python():
    value = os.getenv("SCENE_VALIDATOR_FRAME_PYTHON") or SETTINGS["frame_python"]
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else ROOT / path)


@dataclass(frozen=True)
class VLMSettings:
    base_url: str
    model: str
    timeout: int
    api_key: str = field(repr=False)


def vlm_settings(environ=None):
    """Resolve one explicit endpoint/model/key profile; never borrow DashScope credentials."""
    if environ is None:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        environ = os.environ
    base_url = (environ.get("SCENE_VALIDATOR_BASE_URL") or SETTINGS["base_url"]).rstrip("/")
    model = (environ.get("SCENE_VALIDATOR_MODEL") or SETTINGS["model"]).strip()
    timeout = int(environ.get("SCENE_VALIDATOR_TIMEOUT") or SETTINGS["timeout"])
    if urlsplit(base_url).scheme not in {"http", "https"} or not urlsplit(base_url).netloc:
        raise ValueError("SCENE_VALIDATOR_BASE_URL must be an HTTP(S) URL")
    if not model or timeout <= 0:
        raise ValueError("Visual model and positive timeout are required")
    selected = environ.get("SCENE_VALIDATOR_API_KEY_ENV")
    names = [selected] if selected else [SETTINGS["api_key_env"], *SETTINGS.get("api_key_fallback_envs", [])]
    key = environ.get("SCENE_VALIDATOR_API_KEY", "").strip()
    if not key:
        for name in names:
            key = environ.get(name, "").strip()
            if key:
                break
    if not key:
        raise RuntimeError("Visual evaluator credential missing; set SCENE_VALIDATOR_API_KEY or " + ", ".join(names))
    return VLMSettings(base_url=base_url, model=model, timeout=timeout, api_key=key)
