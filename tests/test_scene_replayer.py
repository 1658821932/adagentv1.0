"""Native replay is exact trajectory execution, with no planner or visual judge."""
import json
import pickle
from pathlib import Path
import pytest
from tools.registry import call_tool
from mcp_servers.worldengine.adapter.replay_native import prepare

@pytest.fixture
def replay_request(tmp_path):
    scene={'length':5,'cameras':{'CAM_F0':{'width':640},'CAM_L0':{'width':640}},'object_track':{'ego':{'state':'original'}}}
    source=tmp_path/'source.pkl';source.write_bytes(pickle.dumps({'scene-key':scene}))
    return dict(scene_id='scene-key',scenario_pkl=str(source),cameras=['CAM_F0'],
        staged_scenario=str(tmp_path/'staged.pkl'),output_dir=str(tmp_path/'output'),asset_folder_path=str(tmp_path/'assets'))

def test_replay_preparation_preserves_source_and_uses_exact_track_controller(replay_request):
    source=Path(replay_request['scenario_pkl']);before=source.read_bytes()
    scene,arguments=prepare(replay_request)
    assert source.read_bytes()==before
    assert list(scene['cameras'])==['CAM_F0']
    assert scene['object_track']['ego']['state']=='original'
    assert 'ego_policy=trajectory_policy' in arguments
    assert 'ego_controller=log_play_controller' in arguments
    assert 'use_planner_actions=false' in arguments
    assert 'with_metric_manager=false' in arguments
    assert 'num_history=4' in arguments and 'num_future=1' in arguments
    assert set(pickle.loads(Path(replay_request['staged_scenario']).read_bytes()))=={'scene-key'}

def test_replay_rejects_a_different_scene_instead_of_falling_back(replay_request):
    replay_request['scene_id']='unknown'
    with pytest.raises(ValueError,match='absent'):prepare(replay_request)

def test_replay_rejects_unbound_camera(replay_request):
    replay_request['cameras']=['invented-camera']
    with pytest.raises(ValueError,match='Unknown replay cameras'):prepare(replay_request)

def test_replay_mock_does_not_claim_real_video():
    result=__import__('tools.tool07_scene_replayer.mock',fromlist=['run']).run(dict(simulation_input=dict(scenario_data_folder_path='/scenario',asset_folder_path='/assets'),scene_id='scene'))
    assert result['status']=='success'
    assert result['result']['mode']=='mock' and result['result']['videos']==[]
    assert result['result']['checks']['native_replay']=='not_run'

def test_missing_native_package_is_not_replaced_by_frozen_metadata(tmp_path,monkeypatch):
    monkeypatch.setenv('ADAGENT_RUNTIME_ROOT',str(tmp_path/'runtime'))
    result=__import__('tools.tool07_scene_replayer.real',fromlist=['run']).run(dict(simulation_input=dict(scenario_data_folder_path=str(tmp_path/'frozen'),asset_folder_path=str(tmp_path/'assets')),scene_id='scene'))
    assert result['status']=='error'
    assert 'Prepared native replay scenario missing' in result['error']['message']

def test_deleted_validator_is_not_available():
    with pytest.raises(ValueError,match='Unknown tool'):call_tool('scene_validator',{},mode='mock')
    assert not Path('tools/tool10_scene_validator').exists()

def test_real_replay_requires_native_report_and_decoded_video(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from mcp_servers.worldengine.adapter import replay
    monkeypatch.setenv('ADAGENT_RUNTIME_ROOT',str(tmp_path/'runtime'))
    source=tmp_path/'native';source.mkdir();(source/'all_scenarios.pkl').write_bytes(b'prepared')
    assets=tmp_path/'assets';assets.mkdir()
    monkeypatch.setattr(replay,'SIMENGINE_PYTHON',Path(__import__('sys').executable))
    def runner(command,**kwargs):
        req=json.loads(Path(command[-1]).read_text())
        output=Path(req['output_dir']);output.mkdir()
        (output/'scene.mp4').write_bytes(b'video')
        Path(req['report_path']).write_text(json.dumps(dict(status='completed',scene_id=req['scene_id'])))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(replay,'run_logged',runner)
    monkeypatch.setattr(replay,'decode_video',lambda path:5)
    result=replay.run_replay(dict(simulation_input=dict(scenario_data_folder_path=str(source),asset_folder_path=str(assets)),scene_id='key'))
    assert result['checks']=={'native_replay':'passed','video_decode':'passed'}
    assert len(result['videos'])==1

def test_real_native_log_length_and_image_saving_are_used(replay_request):
    source=Path(replay_request['scenario_pkl']);scene=pickle.loads(source.read_bytes())['scene-key']
    scene['log_length']=scene.pop('length');source.write_bytes(pickle.dumps({'scene-key':scene}))
    staged,args=prepare(replay_request)
    assert staged['log_length']==5 and 'save_data=true' in args


def test_native_main_creates_output_once_and_reports_exported_videos(replay_request,tmp_path,monkeypatch):
    import sys
    from mcp_servers.worldengine.adapter import replay_native
    request=dict(replay_request,worldengine_root=str(tmp_path),report_path=str(tmp_path/'report.json'))
    path=tmp_path/'request.json';path.write_text(json.dumps(request))
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(sys,'argv',['replay_native','--request',str(path)])
    def run_entry(*args,**kwargs):
        (Path(request['output_dir'])/'runner_report_done.json').write_text(json.dumps([dict(succeeded=True)]))
    monkeypatch.setattr(replay_native.runpy,'run_path',run_entry)
    monkeypatch.setattr(replay_native,'export_videos',lambda output,scene,*args:{'CAM_F0':5})
    replay_native.main()
    report=json.loads(Path(request['report_path']).read_text())
    assert report['status']=='completed' and report['videos']=={'CAM_F0':5}

def test_json_camera_calibration_is_normalized_for_native_mtgs(replay_request):
    import numpy as np
    source=Path(replay_request['scenario_pkl']);data=pickle.loads(source.read_bytes())
    data['scene-key']['cameras']['CAM_F0'].update(sensor2ego_rotation=[1.,0.,0.,0.],sensor2ego_translation=[1.,0.,1.5])
    source.write_bytes(pickle.dumps(data));scene,_=prepare(replay_request)
    camera=scene['cameras']['CAM_F0']
    assert isinstance(camera['sensor2ego_rotation'],np.ndarray) and isinstance(camera['sensor2ego_translation'],np.ndarray)

def test_decode_checks_real_decoded_frame_count(monkeypatch):
    from types import SimpleNamespace
    from mcp_servers.worldengine.adapter import replay
    calls=[]
    def completed(command,**kwargs):
        calls.append(command);return SimpleNamespace(returncode=0,stdout='frame=1\nframe=24\nprogress=end\n')
    monkeypatch.setattr(replay.subprocess,'run',completed)
    assert replay.decode_video(Path('/video.mp4'))==24
    assert '-xerror' in calls[0] and '-progress' in calls[0]
    monkeypatch.setattr(replay.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout='progress=end\n'))
    with pytest.raises(RuntimeError,match='no readable frames'):replay.decode_video(Path('/empty.mp4'))
