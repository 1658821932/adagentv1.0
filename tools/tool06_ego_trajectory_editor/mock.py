"""Interface-only placeholder; no scene or trajectory is generated."""
from mcp_servers.scene_editing.adapter.mock import ego

def run(input_data):
    return {"status":"success","tool":"ego_trajectory_editor","result":ego(input_data)}
