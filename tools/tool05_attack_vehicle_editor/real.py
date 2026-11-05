"""Real stage adapter; disconnected backend is an explicit error."""
from mcp_servers.scene_editing.adapter.service import run_attack_editing, EditingError

def run(input_data):
    try:
        result = run_attack_editing(input_data)
    except EditingError as error:
        return {"status":"error","tool":"attack_vehicle_editor","error":{"type":error.code,"message":str(error)}}
    return {"status":"success","tool":"attack_vehicle_editor","result":result}
