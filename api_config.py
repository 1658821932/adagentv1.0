"""Project-wide API profiles. Settings: configs/api.json; credentials: root .env."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shlex
from urllib.parse import urlsplit

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
                key = key.strip()
                if separator and key and not key.startswith("#"):
                    parts = shlex.split(value, comments=True)
                    os.environ.setdefault(key, " ".join(parts))
    else:
        load_dotenv(ENV_PATH, override=False)


def validate_llm_profile(profile):
    """Validate public connection settings without accepting credential values."""
    if not isinstance(profile, dict):
        raise ValueError("LLM profile must be an object")
    if any(key in profile for key in ("api_key", "authorization", "password", "secret")):
        raise ValueError("Store credentials in .env; profiles only accept api_key_env")
    for field in ("base_url", "model", "api_key_env"):
        if not isinstance(profile.get(field), str) or not profile[field].strip():
            raise ValueError("LLM profile requires " + field)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", profile["api_key_env"]):
        raise ValueError("api_key_env must be an environment variable name, not a key value")
    url = urlsplit(profile["base_url"])
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("base_url must use HTTPS without embedded credentials or query parameters")
    for field in ("timeout", "max_tokens"):
        if type(profile.get(field)) is not int or profile[field] <= 0:
            raise ValueError("LLM profile requires a positive integer " + field)
    if profile.get('image_tool_choice', 'required') not in ('auto', 'required'):
        raise ValueError('image_tool_choice must be auto or required')
    if profile.get('required_tool_thinking', 'preserve') not in ('preserve', 'disabled'):
        raise ValueError('required_tool_thinking must be preserve or disabled')
    from mcp_servers.scene_editing.core.adversarial.llm_request_policy import validate_request_policy
    validate_request_policy(profile)


def get_profile(name="kimi", *, require_key=False, environ=None):
    # "kimi" remains the common reasoning alias for existing clients. The
    # retrieval profile is independent and never participates in this switch.
    profiles = json.loads(API_CONFIG_PATH.read_text())
    reasoning = name == "kimi"
    if reasoning:
        name = profiles.get("active_profile", "kimi")
    if not isinstance(name, str) or name == "active_profile":
        raise ValueError("active_profile must name a saved LLM profile")
    if name not in profiles:
        raise ValueError(f"Unknown API profile: {name}")
    result = deepcopy(profiles[name])
    if reasoning:
        validate_llm_profile(result)
    if environ is None:
        load_credentials()
        environ = os.environ
    if require_key:
        key = environ.get(result["api_key_env"])
        if not key:
            raise RuntimeError(result["api_key_env"] + " is required for API profile " + name)
        result["api_key"] = key
    return result


def wfa_settings():
    profile = get_profile("kimi", require_key=True)
    return {"endpoint": profile["base_url"].rstrip("/") + "/chat/completions",
            "model": profile["model"], "api_key": profile["api_key"],
            "timeout": profile["timeout"], "max_tokens": profile["max_tokens"]}
