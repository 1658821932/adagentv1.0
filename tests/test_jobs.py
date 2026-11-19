"""Execution lifecycle tests use real child processes, without models or GPUs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from agent.jobs import JobStore
from agent.executor import ToolExecutor
from agent.schemas import validate_tool_result

FAKE = """
import json,sys,time
from pathlib import Path
from agent.jobs import locked,write
folder=Path(sys.argv[1])
with locked(folder/'state.lock'):
 m=json.loads((folder/'manifest.json').read_text())
 m['status']='running'
 write(folder/'manifest.json',m)
with (folder.parent/'launches').open('a') as f:f.write('started\\n')
time.sleep(.6)
req=json.loads((folder/'request.json').read_text())
with locked(folder/'state.lock'):
 write(folder/'result.json',dict(status='success',tool=req['tool'],result={'ok':True}))
 m['status']='completed'
 write(folder/'manifest.json',m)
"""

def fake_launcher():
    original = subprocess.Popen
    def launch(command, **kwargs):
        return original([sys.executable, "-c", FAKE, command[-1]], **kwargs)
    return patch("agent.jobs.subprocess.Popen", side_effect=launch)


def test_independent_06_07_background_jobs_are_persistent_and_not_duplicated(tmp_path):
    modes={'ego_trajectory_editor':'mock','policy_runner':'mock'}
    executor=ToolExecutor(modes,jobs_root=tmp_path,execution_timeout=5,scope='pair')
    with fake_launcher():
        expert=executor.execute('ego_trajectory_editor',{'attack_result':'frozen','_execution':{'background':True}})
        policy=executor.execute('policy_runner',{'scene':'frozen','_execution':{'background':True}})
        assert expert['status']==policy['status']=='running'
        first,second=expert['job']['job_id'],policy['job']['job_id']
        assert first!=second
        restored=ToolExecutor(modes,jobs_root=tmp_path,execution_timeout=5,scope='pair')
        assert restored.execute('ego_trajectory_editor',{'_execution':{'job_id':first}})['status']=='success'
        assert restored.execute('policy_runner',{'_execution':{'job_id':second}})['status']=='success'
        assert (tmp_path/'launches').read_text().count('started')==2
        assert restored.execute('policy_runner',{'scene':'frozen','_execution':{'background':True}})['job']['job_id']==second
        assert (tmp_path/'launches').read_text().count('started')==2

def test_timeout_concurrent_submit_and_restart_reuse(tmp_path):
    store=JobStore(tmp_path)
    with fake_launcher():
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids=list(pool.map(lambda _:store.submit("test","mock",{"x":1},"run"),range(4)))
        assert len(set(ids))==1
        job=ids[0]
        pending=store.wait(job,.01)
        assert pending["status"]=="running"
        assert validate_tool_result(pending,"test")[0]
        assert store.submit("test","mock",{"x":1},"run",retry=True)==job
        assert store.wait(job,5)["status"]=="success"
        # New store instance can recover and reuse the persisted job.
        assert JobStore(tmp_path).submit("test","mock",{"x":1},"run")==job
        assert (tmp_path/"launches").read_text().count("started")==1
        again=store.submit("test","mock",{"x":1},"run",retry=True)
        assert again!=job
        assert store.wait(again,5)["status"]=="success"
        other=store.submit("test","mock",{"x":1},"other-run")
        assert other!=again
        assert store.wait(other,5)["status"]=="success"

def test_failed_job_requires_explicit_retry(tmp_path):
    store=JobStore(tmp_path)
    # Actual production worker, unknown tool fails without network.
    job=store.submit("unknown","mock",{},"run")
    assert store.wait(job,10)["status"]=="error"
    assert store.submit("unknown","mock",{},"run")==job
    retry=store.submit("unknown","mock",{},"run",retry=True)
    assert retry!=job
    assert store.wait(retry,10)["status"]=="error"

def test_worker_death_is_not_left_running(tmp_path):
    store=JobStore(tmp_path)
    with fake_launcher():
        job=store.submit("test","mock",{},"run")
        manifest=json.loads((tmp_path/job/"manifest.json").read_text())
        os.kill(manifest["pid"],signal.SIGKILL)
        assert store.wait(job,5)["error"]["type"]=="job_interrupted"

def test_interrupted_ego_job_exposes_matching_preparation_run(tmp_path,monkeypatch):
    from agent.jobs import preparation_run_dir
    monkeypatch.setattr('agent.jobs.PROJECT_ROOT',tmp_path)
    runtime=tmp_path/'mcp_servers/scene_editing/runtime/run_test'
    runtime.mkdir(parents=True)
    (runtime/'manifest.json').write_text(json.dumps(dict(operation='ego_editing',owner_pid=123,run_dir=str(runtime))))
    assert preparation_run_dir(dict(tool='ego_trajectory_editor',pid=123))==str(runtime)
    assert preparation_run_dir(dict(tool='attack_vehicle_editor',pid=123)) is None
    assert preparation_run_dir(dict(tool='ego_trajectory_editor',pid=999)) is None


def test_actual_executor_mock_job_and_query(tmp_path):
    executor=ToolExecutor({"failure_analyzer":"mock"},jobs_root=tmp_path,execution_timeout=10,scope="one")
    result=executor.execute("failure_analyzer",{"token":"example"})
    assert result["status"]=="success"
    job=result["job"]["job_id"]
    assert executor.execute("failure_analyzer",{"token":"example"})["job"]["job_id"]==job
    assert executor.execute("failure_analyzer",{"_execution":{"job_id":job}})["status"]=="success"
    other=ToolExecutor({"failure_analyzer":"mock"},jobs_root=tmp_path,scope="other")
    assert other.execute("failure_analyzer",{"_execution":{"job_id":job}})["status"]=="error"


def test_orphan_child_blocks_replacement(tmp_path):
    store=JobStore(tmp_path)
    with fake_launcher():
        job=store.submit("test","mock",{},"run")
        manifest=json.loads((tmp_path/job/"manifest.json").read_text())
        child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(15)"],start_new_session=True)
        try:
            from agent.jobs import write, process_identity
            write(tmp_path/job/"children.json",[dict(pid=child.pid,process_start=process_identity(child.pid))])
            os.kill(manifest["pid"],signal.SIGKILL)
            import time
            for _ in range(100):
                state=store.query(job)
                if state.get("job",{}).get("state")=="orphaned":
                    break
                time.sleep(.01)
            assert state["job"]["state"]=="orphaned"
            assert store.wait(job, 720)['job']['state'] == 'orphaned'
            assert store.submit("test","mock",{},"run",retry=True)==job
        finally:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait()
        assert store.wait(job,5)["error"]["type"]=="job_interrupted"


def test_stale_worker_and_child_identities_cannot_signal_reused_process_groups(tmp_path):
    from agent.jobs import process_record, write

    store = JobStore(tmp_path)
    job = 'a' * 32
    folder = tmp_path / job
    folder.mkdir()
    with stubborn_session(tmp_path, 'unrelated') as (leader, child):
        stale = process_record(leader.pid)
        stale['process_start'] = str(int(stale['process_start']) - 100)
        manifest = dict(stale, status='running', tool='test')
        write(folder / 'manifest.json', manifest)
        write(folder / 'children.json', [stale])
        with patch('agent.jobs.signal.pidfd_send_signal') as kill:
            assert store.query(job)['job']['state'] == 'interrupted'
            write(folder / 'manifest.json', manifest)
            assert store.stop(job)['status'] == 'interrupted'
        kill.assert_not_called()


def stubborn_command(ready):
    child_code = ('import os, signal, time; from pathlib import Path; '
                  'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                  f'Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)')
    leader_code = f'import subprocess, sys; subprocess.Popen([sys.executable,"-c",{child_code!r}]).wait()'
    return [sys.executable, '-c', leader_code]


@contextmanager
def stubborn_session(tmp_path, job_id):
    ready = tmp_path / 'stubborn.pid'
    leader = subprocess.Popen(stubborn_command(ready), start_new_session=True,
                              env={**os.environ, 'ADAGENT_JOB_ID': job_id})
    try:
        deadline = time.monotonic() + 5
        while not (ready.exists() and ready.stat().st_size) and time.monotonic() < deadline:
            time.sleep(.01)
        yield leader, int(ready.read_text())
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        leader.wait(timeout=5)


def test_stop_kills_sigterm_resistant_descendant_after_leader_exits(tmp_path):
    from agent.jobs import process_identity, process_record, write

    store = JobStore(tmp_path)
    job = 'a'*32
    folder = tmp_path / job
    folder.mkdir()
    with stubborn_session(tmp_path, 'unmarked') as (leader, child), ThreadPoolExecutor() as pool:
        write(folder / 'manifest.json', dict(process_record(leader.pid), tool='test', status='running'))
        assert process_identity(child)
        reaped = pool.submit(leader.wait, 5)
        assert store.stop(job)['status'] == 'interrupted'
        assert reaped.result() == -signal.SIGTERM
        assert process_identity(child) is None


def test_orphaned_descendant_without_worker_or_child_session_leader_blocks_retry(tmp_path):
    from agent.jobs import process_identity, process_record, write

    store = JobStore(tmp_path)
    with fake_launcher():
        job = store.submit('test', 'mock', {}, 'run')
    manifest = json.loads((tmp_path / job / 'manifest.json').read_text())
    os.kill(manifest['pid'], signal.SIGKILL)
    with stubborn_session(tmp_path, job) as (leader, child):
        write(tmp_path / job / 'children.json', [process_record(leader.pid)])
        leader.terminate()
        assert leader.wait(timeout=5) == -signal.SIGTERM
        assert process_identity(child)
        assert store.wait(job, 5)['job']['state'] == 'orphaned'
        with fake_launcher():
            assert store.submit('test', 'mock', {}, 'run', retry=True) == job
        assert store.stop(job)['status'] == 'interrupted'
        assert process_identity(child) is None


def test_subprocess_registration_and_real_timeout_cleanup(tmp_path,monkeypatch):
    import pytest
    from mcp_servers.process import run_logged
    from agent.jobs import process_identity
    monkeypatch.setenv("ADAGENT_JOB_DIR",str(tmp_path))
    ready = tmp_path / 'stubborn.pid'
    with pytest.raises(subprocess.TimeoutExpired):
        run_logged(stubborn_command(ready), cwd=tmp_path, log_path=tmp_path/'child.log', timeout=1.)
    children=json.loads((tmp_path/"children.json").read_text())
    assert len(children)==1
    assert process_identity(children[0]["pid"]) is None
    assert process_identity(int(ready.read_text())) is None


def test_stop_scope_kills_worker_and_registered_child_groups(tmp_path):
    from agent.jobs import write, process_identity
    store=JobStore(tmp_path)
    with fake_launcher():
        job=store.submit("test","mock",{},"run")
        manifest=json.loads((tmp_path/job/"manifest.json").read_text())
        child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(15)"],start_new_session=True)
        write(tmp_path/job/"children.json",[dict(pid=child.pid,process_start=process_identity(child.pid))])
        try:
            assert store.stop_scope("run")==[job]
            assert process_identity(manifest["pid"]) is None
            assert process_identity(child.pid) is None
            stopped=json.loads((tmp_path/job/"manifest.json").read_text())
            assert stopped["status"]=="interrupted" and stopped["finished_at"] is not None
            result=json.loads((tmp_path/job/"result.json").read_text())
            assert result["error"]["type"]=="job_interrupted"
            assert store.query(job)["status"]=="error"
        finally:
            for pid in (manifest["pid"],child.pid):
                try: os.killpg(pid,signal.SIGKILL)
                except ProcessLookupError: pass
            child.wait()


def test_stop_scope_does_not_touch_completed_or_other_scope_jobs(tmp_path):
    store=JobStore(tmp_path)
    with fake_launcher():
        completed=store.submit("test","mock",{},"done")
        other=store.submit("test","mock",{},"other")
        assert store.stop_scope("other")==[other]
        assert store.wait(other,5)["error"]["type"]=="job_interrupted"
        assert store.stop_scope("other")==[]
        assert store.wait(completed,5)["status"]=="success"
        assert store.stop_scope("done")==[]
        assert json.loads((tmp_path/completed/"manifest.json").read_text())["status"]=="completed"
