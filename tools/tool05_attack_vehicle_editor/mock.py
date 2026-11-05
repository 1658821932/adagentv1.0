"""Interface-only placeholder; no scene or trajectory is generated."""
from mcp_servers.scene_editing.adapter.mock import attack

def run(input_data):
    return {"status":"success","tool":"attack_vehicle_editor","result":attack(input_data)}
