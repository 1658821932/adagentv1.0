"""Standalone API profiles with optional host injection; no host imports."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import json
import os
from pathlib import Path

_PROVIDER = ContextVar("api_profile_provider", default=None)
_REQUEST_AUDIT = ContextVar('request_audit', default=None)
ENV_NAME = "SENTRY_API_CONFIG"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "api.example.json"

@contextmanager
def bind_profiles(provider, *, request_audit=None):
    token = _PROVIDER.set(provider)
    audit_token = _REQUEST_AUDIT.set(request_audit)
    try:
        yield
    finally:
        _REQUEST_AUDIT.reset(audit_token)
        _PROVIDER.reset(token)


def request_json(call, payload, *, endpoint, label, secrets):
    audit = _REQUEST_AUDIT.get()
    if audit is not None:
        return audit(call, payload, endpoint=endpoint, label=label, secrets=secrets, return_response=True)
    response = call()
    return response, response.json()

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

def vlm_settings(config_path=None):
    p = get_profile("kimi",require_key=True,config_path=config_path)
    return {"endpoint":p["base_url"].rstrip("/")+"/chat/completions","model":p["model"],"api_key":p["api_key"]}
