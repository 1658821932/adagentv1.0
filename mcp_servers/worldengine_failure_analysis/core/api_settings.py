"""Standalone API profiles with optional host injection; no host imports."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import json
import os
from pathlib import Path

_PROVIDER = ContextVar("api_profile_provider", default=None)
ENV_NAME = "WFA_API_CONFIG"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "api.example.json"

@contextmanager
def bind_profiles(provider):
    token = _PROVIDER.set(provider)
    try:
        yield
    finally:
        _PROVIDER.reset(token)

def get_profile(name="kimi", *, require_key=False, environ=None, config_path=None):
    provider = _PROVIDER.get()
    if provider is not None and config_path is None:
        return provider(name, require_key=require_key, environ=environ)
    env = os.environ if environ is None else environ
    path = Path(config_path or env.get(ENV_NAME) or DEFAULT_CONFIG).expanduser()
    profile = deepcopy(json.loads(path.read_text())[name])
    if name == "kimi":
        profile["base_url"] = env.get("OPENAI_BASE_URL") or env.get("LLM_BASE_URL") or profile["base_url"]
        profile["model"] = env.get("OPENAI_MODEL") or env.get("LLM_MODEL") or profile["model"]
    if require_key:
        key = env.get(profile["api_key_env"]) or (env.get("LLM_API_KEY") if name == "kimi" else None)
        if not key:
            raise RuntimeError(profile["api_key_env"] + " is required for API profile " + name)
        profile["api_key"] = key
    return profile

def vlm_settings(config_path=None):
    p = get_profile("kimi",require_key=True,config_path=config_path)
    return {"endpoint":p["base_url"].rstrip("/")+"/chat/completions","model":p["model"],"api_key":p["api_key"]}
