"""SmartEdit integration paths; credentials remain in local environment files."""

import os
import json
from pathlib import Path
from mcp_servers.configuration import runtime_path, settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_ROOT = Path(__file__).resolve().parents[1]
SMART_EDIT_ROOT = Path(os.environ.get("SMART_EDIT_ROOT") or COMPONENT_ROOT / "core").expanduser().resolve()
SMART_EDIT_PYTHON = runtime_path(COMPONENT_ROOT, "python", "SMART_EDIT_PYTHON", "environments/smart-edit/.venv/bin/python")
SMART_EDIT_CLIENT = Path(__file__).resolve().parent / "client.py"


def llm_settings() -> dict:
    """Resolve model settings without requiring shell exports for JSON config."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    defaults = settings(COMPONENT_ROOT).get("llm", {})
    role = {}
    return {
        "api_key": os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or defaults.get("base_url"),
        "model": role.get("model") or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or defaults.get("model"),
        "max_tokens": int(role.get("max_tokens") or os.getenv("OPENAI_MAX_TOKENS") or defaults.get("max_tokens", 2048)),
        "timeout": int(role.get("timeout") or os.getenv("LLM_TIMEOUT") or os.getenv("OPENAI_TIMEOUT") or defaults.get("timeout", 600)),
    }
