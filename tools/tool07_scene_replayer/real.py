from mcp_servers.worldengine.adapter.service import run_replay

def run(input_data):
    try:
        result=run_replay(input_data)
    except (OSError,ValueError,RuntimeError) as error:
        return dict(status='error',tool='scene_replayer',error=dict(type='replay_execution_error',message=str(error)[:1000]))
    return dict(status='success',tool='scene_replayer',result=result)
