import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.tool06_reconstruction.real import run


def test_manifest_dispatch_uses_native_scene_result_instead_of_dggt_ply():
    expected={'backend':'adscenegraph','checkpoint_path':'/new/checkpoint_final.pth','rendered_video':'/new/rendered.mp4'}
    proc=SimpleNamespace(returncode=0,stdout=json.dumps({'status':'success','result':expected}),stderr='')
    with patch('mcp_servers.dggt.adapter.client.Path.is_file',return_value=True),patch('mcp_servers.dggt.adapter.client.subprocess.run',return_value=proc) as execute:
        result=run({'video_path':'/new/video.mp4','scene_manifest_path':'/new/scene_manifest.json'})
    assert result['result']==expected
    assert 'scene_ply' not in result['result']
    arguments=execute.call_args
    assert 'mcp_servers.nuplan_data.reconstruct' in arguments.args[0]
    assert json.loads(arguments.kwargs['input'])['scene_manifest_path']=='/new/scene_manifest.json'


def test_annotation_backend_requires_a_manifest():
    with pytest.raises(ValueError,match='requires scene_manifest_path'):
        run({'backend':'adscenegraph','video_path':'/new/video.mp4'})


def test_dggt_still_enforces_its_own_frame_limit():
    with pytest.raises(ValueError,match='num_frames=40'):
        run({'backend':'dggt','video_path':'/new/video.mp4','num_frames':100})


def test_native_backend_failure_is_not_reported_as_success():
    proc=SimpleNamespace(returncode=1,stdout=json.dumps({'status':'error','error':{'message':'input sensor file is missing'}}),stderr='')
    with patch('mcp_servers.dggt.adapter.client.Path.is_file',return_value=True),patch('mcp_servers.dggt.adapter.client.subprocess.run',return_value=proc):
        with pytest.raises(RuntimeError,match='sensor file is missing'):
            run({'backend':'adscenegraph','scene_manifest_path':'/new/manifest.json'})


def test_missing_native_runtime_fails_before_launch(tmp_path):
    with patch('mcp_servers.nuplan_data.common.RECONSTRUCTION_PYTHON',tmp_path/'missing-python'), \
         patch('mcp_servers.dggt.adapter.client.subprocess.run') as execute:
        with pytest.raises(RuntimeError,match='environment not found'):
            run({'backend':'adscenegraph','scene_manifest_path':'/manifest.json'})
    execute.assert_not_called()


def test_configured_interpreter_keeps_venv_prefix(tmp_path,monkeypatch):
    import subprocess
    import venv
    from pathlib import Path
    from mcp_servers.nuplan_data.common import configured_path
    environment=tmp_path/'venv'
    venv.EnvBuilder(with_pip=False,symlinks=True).create(environment)
    interpreter=environment/'bin/python'
    monkeypatch.setenv('TEST_ADSCENEGRAPH_PYTHON',str(interpreter))
    result=configured_path('test_interpreter','TEST_ADSCENEGRAPH_PYTHON',interpreter,follow_symlinks=False)
    prefix=subprocess.check_output([str(result),'-c','import sys; print(sys.prefix)'],text=True).strip()
    assert Path(prefix)==environment
