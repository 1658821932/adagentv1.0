"""GL library paths must reach both native processes, including direct launch."""
import argparse
import os
from pathlib import Path

from mcp_servers.worldengine.adapter import config
from mcp_servers.worldengine.adapter.launch import build_commands


def test_runtime_env_preserves_overrides_trace_and_existing_paths_without_mutation(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'SHARED_LIBRARY_DIR',tmp_path/'unused-default')
    original=dict(WORLDENGINE_LIBRARY_DIR=str(tmp_path),LD_LIBRARY_PATH='/existing/lib',
                  ADAGENT_AGENT_RUN_ID='fixture',CUDA_VISIBLE_DEVICES='fixture-device')
    actual=config.runtime_env(original)
    assert actual['LD_LIBRARY_PATH']==str(tmp_path)+os.pathsep+'/existing/lib'
    assert original['LD_LIBRARY_PATH']=='/existing/lib'
    assert actual['CUDA_VISIBLE_DEVICES']=='fixture-device' and actual['ADAGENT_AGENT_RUN_ID']=='fixture'
    assert config.runtime_env(actual)==actual


def test_direct_launcher_uses_configured_shared_libraries_for_both_children(tmp_path,monkeypatch):
    library=tmp_path/'libraries';library.mkdir()
    monkeypatch.setattr(config,'SHARED_LIBRARY_DIR',library)
    args=argparse.Namespace(config='/fixture/config',checkpoint='/fixture/weights',model_name='fixture',
        data_type='fixture',react_type='NR',asset_name='fixture',added_name='fixture')
    source=dict(WORLDENGINE_ROOT=str(tmp_path),WORLDENGINE_TEST_PATH=str(tmp_path/'out'),
        LD_LIBRARY_PATH='/existing/lib',NUM_HISTORY='4',MAX_FRAME_PER_SCENE='16',WITH_DENSE_REWARD_MANAGER='true')
    sim,alg,_,_,_,env,effective=build_commands(args,source)
    assert env['LD_LIBRARY_PATH']==str(library)+os.pathsep+'/existing/lib'
    assert env['WORLDENGINE_LIBRARY_DIR']==str(library)
    assert effective['native_libraries']['directory']==str(library)
    assert 'with_dense_reward_manager=true' in sim
    assert effective['max_frame_per_scene']==16
    assert source['LD_LIBRARY_PATH']=='/existing/lib'


def test_runtime_env_does_not_invent_an_unavailable_library_directory(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'SHARED_LIBRARY_DIR',tmp_path/'missing')
    assert config.runtime_env(dict(LD_LIBRARY_PATH='/existing'))==dict(LD_LIBRARY_PATH='/existing')
