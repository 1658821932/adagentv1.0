"""Mock replay returns no actual videos or accepted artifacts."""
def run(input_data):
    return dict(status='success',tool='scene_replayer',result=dict(mode='mock',scene_id=input_data['scene_id'],
        result_dir='/mock/replay',videos=[],replay_report='/mock/replay/report.json',
        checks={'native_replay':'not_run','video_decode':'not_run'}))
