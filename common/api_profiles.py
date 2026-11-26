"""Per-tool LLM profile and thinking-level selection.

Every tool that calls a model reads its settings from one place: configs/api.json
under "tool_profiles" (which API profile) and "tool_efforts" (how hard to think).
"active_profile" means follow the main agent; anything else pins that tool.

A tool effort of "active_profile" follows the global "reasoning_effort" entry, so a
single switch still moves everything that was not pinned.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_CONFIG = PROJECT_ROOT / "configs/api.json"
DEFAULT = "active_profile"
LEVELS = ("low", "medium", "high")

ENV = {
    "failure_analyzer": "FAILURE_ANALYZER_LLM_PROFILE",
    "scene_instance_generator": "SCENE_INSTANCE_GENERATOR_LLM_PROFILE",
    "edit_planner": "EDIT_PLANNER_LLM_PROFILE",
    "attack_editor": "ATTACK_EDITOR_LLM_PROFILE",
    "ego_editor": "EGO_EDITOR_LLM_PROFILE",
}
EFFORT_ENV = {
    "failure_analyzer": "FAILURE_ANALYZER_REASONING_EFFORT",
    "scene_instance_generator": "SCENE_INSTANCE_GENERATOR_REASONING_EFFORT",
    "edit_planner": "EDIT_PLANNER_REASONING_EFFORT",
    "attack_editor": "ATTACK_EDITOR_REASONING_EFFORT",
    "ego_editor": "EGO_EDITOR_REASONING_EFFORT",
}


def _config() -> dict:
    try:
        return json.loads(API_CONFIG.read_text())
    except (OSError, ValueError):
        return {}


def selections() -> dict:
    data = _config()
    configured = data.get("tool_profiles") or {}
    # scene_editing is the legacy key that used to cover both editors.
    legacy = configured.get("scene_editing", DEFAULT)
    result = {}
    for component in ENV:
        fallback = legacy if component in ("attack_editor", "ego_editor") else DEFAULT
        result[component] = configured.get(component, fallback)
    return result


def tool_profile(component: str) -> str:
    """Profile name for one tool; 'active_profile' means follow the main agent."""
    return os.getenv(ENV[component]) or selections().get(component, DEFAULT)


def profile_argument(component: str) -> str:
    """Argument for get_profile(): 'kimi' is the historical alias for active_profile."""
    name = tool_profile(component)
    if name == DEFAULT:
        return "kimi"
    if name == "kimi" and "kimi_k3" in _config():
        return "kimi_k3"
    return name


def effort_selections() -> dict:
    configured = _config().get("tool_efforts") or {}
    return {component: configured.get(component, DEFAULT) for component in ENV}


def effort_for(component: str):
    """Thinking level pinned to this tool, or None to follow the global setting."""
    override = os.getenv(EFFORT_ENV[component])
    if override in LEVELS:
        return override
    value = effort_selections().get(component, DEFAULT)
    return value if value in LEVELS else None


def reasoning_effort_level():
    """Global thinking level; None keeps each profile's own value."""
    override = os.getenv("LLM_REASONING_EFFORT")
    if override in LEVELS:
        return override
    value = _config().get("reasoning_effort")
    return value if value in LEVELS else None
