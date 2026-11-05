"""Non-secret stage commands; empty means the real backend is not connected."""
from pathlib import Path
from mcp_servers.configuration import settings

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = COMPONENT_ROOT.parents[1]

def stage_command(stage):
    command = settings(COMPONENT_ROOT).get("commands", {}).get(stage)
    if not isinstance(command, list) or not command or not all(isinstance(s, str) and s.strip() for s in command):
        return None
    return command
