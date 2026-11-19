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
    profiles = json.loads(path.read_text())
    unified = "active_profile" in profiles
    selected = profiles.get("active_profile", "kimi") if name == "kimi" else name
    if not isinstance(selected, str) or selected == "active_profile" or selected not in profiles:
        raise ValueError("Unknown API profile selection")
    profile = deepcopy(profiles[selected])
    if name == "kimi" and not unified:
        profile["base_url"] = env.get("OPENAI_BASE_URL") or env.get("LLM_BASE_URL") or profile["base_url"]
        profile["model"] = env.get("OPENAI_MODEL") or env.get("LLM_MODEL") or profile["model"]
    if require_key:
        key = env.get(profile["api_key_env"]) or (env.get("LLM_API_KEY") if name == "kimi" and not unified else None)
        if not key:
            raise RuntimeError(profile["api_key_env"] + " is required for API profile " + name)
        profile["api_key"] = key
    return profile

LEVEL_TO_EFFORT = {'thinking': {'low': 'low', 'medium': 'high', 'high': 'max'},
                   'qwen': {'low': 'low', 'medium': 'medium', 'high': 'xhigh'},
                   'kimi': {'low': 'low', 'medium': 'high', 'high': 'max'}}


def vlm_settings(config_path=None):
    # The adapter names the profile; "kimi" stays the alias for active_profile.
    p = get_profile(os.environ.get("WFA_LLM_PROFILE") or "kimi",require_key=True,config_path=config_path)
    dialect = p.get("request_dialect")
    effort = p.get("reasoning_effort")
    level = os.environ.get("WFA_REASONING_EFFORT")
    if level in ("low", "medium", "high") and dialect in LEVEL_TO_EFFORT:
        effort = LEVEL_TO_EFFORT[dialect][level]
    return {"endpoint":p["base_url"].rstrip("/")+"/chat/completions","model":p["model"],
            "api_key":p["api_key"],"timeout":p["timeout"],"max_tokens":p["max_tokens"],
            "reasoning_effort":effort,"request_dialect":dialect,"thinking_mode":p.get("thinking_mode")}
