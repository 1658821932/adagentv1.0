"""Project-wide API profiles. Settings: configs/api.json; credentials: root .env."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex

PROJECT_ROOT = Path(__file__).resolve().parent
API_CONFIG_PATH = PROJECT_ROOT / "configs/api.json"
ENV_PATH = PROJECT_ROOT / ".env"


def load_credentials():
    try:
        from dotenv import load_dotenv
    except ImportError:
        # WFA's scientific environment need not install python-dotenv; keys are plain env assignments.
        if ENV_PATH.is_file():
            for line in ENV_PATH.read_text().splitlines():
                text = line.strip()
                if text.startswith("export "):
                    text = text[7:]
                key, separator, value = text.partition("=")
                if separator and key and not key.startswith("#"):
                    parts = shlex.split(value, comments=True)
                    os.environ.setdefault(key, " ".join(parts))
    else:
        load_dotenv(ENV_PATH, override=False)


def get_profile(name="kimi", *, require_key=False, environ=None):
    profiles = json.loads(API_CONFIG_PATH.read_text())
    if name not in profiles:
        raise ValueError(f"Unknown API profile: {name}")
    result = deepcopy(profiles[name])
    if environ is None:
        load_credentials()
        environ = os.environ
    if name == "kimi":
        result["base_url"] = result["base_url"]
        result["model"] = result["model"]
    if require_key:
        key = environ.get(result["api_key_env"])
        if not key:
            raise RuntimeError(result["api_key_env"] + " is required for API profile " + name)
        result["api_key"] = key
    return result


def wfa_settings():
    profile = get_profile("kimi", require_key=True)
    return {"endpoint": profile["base_url"].rstrip("/") + "/chat/completions",
            "model": profile["model"], "api_key": profile["api_key"]}
