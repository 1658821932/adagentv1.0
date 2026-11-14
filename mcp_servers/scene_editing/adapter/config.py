"""Non-secret stage commands; empty means the real backend is not connected."""
from pathlib import Path
from mcp_servers.configuration import settings

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = COMPONENT_ROOT.parents[1]

def stage_command(stage):
    command = settings(COMPONENT_ROOT).get("commands", {}).get(stage)
    if not isinstance(command, list) or not command or not all(isinstance(s, str) and s.strip() for s in command):
        return None
    # Bundled commands may use project-relative paths, including when the
    # caller's working directory is elsewhere. Keep venv interpreter symlinks.
    return [str(PROJECT_ROOT / part) if part.startswith(("environments/", "mcp_servers/"))
            else part for part in command]


def public_pipeline_root():
    value = settings(COMPONENT_ROOT).get("public_pipeline_root")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

def llm_profile_name():
    """Profile argument for get_profile(); follows the configured per-tool selection."""
    from common.api_profiles import profile_argument
    return profile_argument("scene_editing")
