"""Real stage adapter; disconnected backend is an explicit error."""
from mcp_servers.scene_editing.adapter.service import run_ego_editing, EditingError

def run(input_data):
    try:
        result = run_ego_editing(input_data)
    except EditingError as error:
        return {"status":"error","tool":"ego_trajectory_editor","error":{"type":error.code,"message":str(error)}}
    return {"status":"success","tool":"ego_trajectory_editor","result":result}
