"""Real lightweight workers test IPC/reuse/cancellation without loading CUDA or VAD."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import pytest
from mcp_servers.worldengine.adapter import planner_prewarm as warm
from mcp_servers.worldengine.adapter.launch import build_commands


@pytest.fixture
def worker(tmp_path,monkeypatch):
    monkeypatch.setenv('ALGENGINE_PYTHON',sys.executable)
    monkeypatch.setenv('ADAGENT_VAD_PREWARM_ENABLED','true')
    monkeypatch.setenv('FAKE_API_KEY','fixture-secret-not-in-worker')
    root=tmp_path/'engine';entry=root/'projects/AlgEngine/closed_loop/sim_test.py'
    entry.parent.mkdir(parents=True)
    entry.write_text('''import json,os,sys,time
from pathlib import Path
class AsyncFileMonitor:
    def __init__(self,**kwargs):self.options=kwargs
def main():
    output=Path(sys.argv[sys.argv.index('--log-dir')+1]);output.mkdir(parents=True,exist_ok=True)
    values=dict(pid=os.getpid(),credential_present=bool(os.getenv('FAKE_API_KEY')),
        frame_count=AsyncFileMonitor(folder_path='fixture').options['max_frame_per_scene'])
    (output/'job.json').write_text(json.dumps(values))
    if '--fail' in sys.argv:raise RuntimeError('fixture failure')
    if '--hang' in sys.argv:time.sleep(30)
''')
    # Unix sockets have a short path limit, independent of pytest's nested paths.
    with tempfile.TemporaryDirectory(prefix='adagent-warm-test-') as directory:
        folder=Path(directory)
        monkeypatch.setattr(warm,'_layout',lambda policy:(root,entry,Path(sys.executable),folder))
        policy={'worldengine_root':str(root)}
        result=warm.start(policy,tmp_path/'prewarm.json')
        assert result['status']=='warming'
        end=time.monotonic()+10
        while time.monotonic()<end:
            state=warm._read(folder/'state.json')
            if state.get('status') in ('ready','failed'):break
            time.sleep(.02)
        assert state['status']=='ready',(state,(folder/'worker.log').read_text())
        yield policy,entry,folder,result
        state=warm._read(folder/'state.json')
        if warm._alive(state):
            try:
                with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
                    connection.settimeout(.5);connection.connect(str(folder/'worker.sock'))
                    warm._send(connection,{'op':'shutdown'});warm._receive(connection)
            except (OSError,EOFError):
                os.kill(state['pid'],15)
        end=time.monotonic()+3
        while warm._alive(warm._read(folder/'state.json')) and time.monotonic()<end:time.sleep(.02)


def original(entry,output,*extra):
    return [sys.executable,'unused-native-wrapper.py','--scene-frames','16',str(entry),'config.py','weights.pth',
        '--log-dir',str(output),*extra]


def client_command(folder,command):
    return [sys.executable,str(Path(warm.__file__).resolve()),'client','--socket',str(folder/'worker.sock'),
            '--ready-timeout','5','--',*command]


def test_worker_reuses_framework_process_and_separates_jobs(worker,tmp_path):
    policy,entry,folder,first=worker
    second=warm.start(policy,tmp_path/'second-prewarm.json')
    assert second['reused'] is True and first['pid']==second['pid']
    for name in ('one','two'):
        warm.run_client(folder/'worker.sock',original(entry,tmp_path/name))
    one=json.loads((tmp_path/'one/job.json').read_text())
    two=json.loads((tmp_path/'two/job.json').read_text())
    assert one['pid']==two['pid']==first['pid']
    assert one['frame_count']==two['frame_count']==16
    assert one['credential_present'] is two['credential_present'] is False
    assert warm._read(folder/'state.json')['completed_jobs']==2
    assert (folder/'worker.sock').stat().st_mode&0o777==0o600


def test_native_failure_after_acceptance_is_not_replayed_cold(worker,tmp_path):
    _,entry,folder,_=worker
    with pytest.raises(RuntimeError,match='Prewarmed native planner failed'):
        warm.run_client(folder/'worker.sock',original(entry,tmp_path/'failed','--fail'))
    assert (tmp_path/'failed/job.json').exists()
    end=time.monotonic()+3
    while warm._alive(warm._read(folder/'state.json')) and time.monotonic()<end:time.sleep(.02)
    assert warm._read(folder/'state.json')['status']=='failed'


def test_cancelled_client_terminates_active_native_worker(worker,tmp_path):
    _,entry,folder,_=worker
    process=subprocess.Popen(client_command(folder,original(entry,tmp_path/'hanging','--hang')))
    try:
        end=time.monotonic()+5
        while not (tmp_path/'hanging/job.json').exists() and time.monotonic()<end:time.sleep(.02)
        assert (tmp_path/'hanging/job.json').exists()
        process.terminate();process.wait(timeout=3)
        end=time.monotonic()+3
        while warm._alive(warm._read(folder/'state.json')) and time.monotonic()<end:time.sleep(.02)
        assert not warm._alive(warm._read(folder/'state.json'))
    finally:
        if process.poll() is None:process.kill();process.wait()


def test_unavailable_worker_uses_original_cold_command(tmp_path):
    script=tmp_path/'cold.py';script.write_text('from pathlib import Path\nPath(__file__).with_suffix(".ran").touch()\n')
    result=subprocess.run(client_command(tmp_path,[sys.executable,str(script)]),capture_output=True,timeout=5)
    assert result.returncode==0,result.stderr
    assert script.with_suffix('.ran').exists()


def test_disable_and_missing_entry_preserve_cold_launch(tmp_path,monkeypatch):
    monkeypatch.setenv('ADAGENT_VAD_PREWARM_ENABLED','false')
    assert warm.start({},tmp_path/'off.json')['status']=='disabled'
    assert warm.locate({}) is None
    monkeypatch.setenv('ADAGENT_VAD_PREWARM_ENABLED','true')
    assert warm.start({'worldengine_root':str(tmp_path)},tmp_path/'missing.json')['status']=='unavailable'
    assert warm.locate({'worldengine_root':str(tmp_path)}) is None


def test_preheated_launch_preserves_exact_native_arguments(tmp_path):
    args=argparse.Namespace(config='/cfg',checkpoint='/weights',model_name='p',data_type='s',
        react_type='NR',asset_name='s',added_name='s')
    env=dict(WORLDENGINE_ROOT=str(tmp_path),WORLDENGINE_TEST_PATH=str(tmp_path/'out'))
    _,cold,*_,normal=build_commands(args,env)
    _,actual,*_,effective=build_commands(args,dict(env,ADAGENT_VAD_PREWARM_SOCKET='/tmp/private.sock'))
    assert actual[actual.index('--')+1:]==cold
    assert effective['planner']==normal['planner']
    assert effective['evaluation_horizon']==normal['evaluation_horizon']
    assert effective['planner_prewarm']['scope']=='framework_imports_only_fresh_model_per_candidate'
