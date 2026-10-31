"""Actual subprocess tests for live logs, timeout cleanup and trace propagation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from mcp_servers.process import run_logged
from mcp_servers.runtime import runtime_task,current_run,trace_context

def test_log_is_visible_before_process_completes(tmp_path):
    log=tmp_path/"live.log"
    command=[sys.executable,"-u","-c",'import time; print("first",flush=True); time.sleep(0.4); print("last",flush=True)']
    with ThreadPoolExecutor() as executor:
        pending=executor.submit(run_logged,command,cwd=tmp_path,log_path=log,timeout=5)
        deadline=time.monotonic()+3
        while time.monotonic()<deadline and (not log.exists() or "first" not in log.read_text()):
            time.sleep(0.01)
        assert "first" in log.read_text() and not pending.done()
        assert pending.result().returncode==0
    assert "last" in log.read_text()

def test_timeout_kills_grandchildren_and_preserves_progress(tmp_path):
    pidfile=tmp_path/"child.pid";log=tmp_path/"timeout.log"
    code='import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); Path(sys.argv[1]).write_text(str(p.pid)); print("started",flush=True); time.sleep(60)'
    with pytest.raises(subprocess.TimeoutExpired):
        run_logged([sys.executable,"-u","-c",code,str(pidfile)],cwd=tmp_path,log_path=log,timeout=0.5)
    assert "started" in log.read_text()
    pid=int(pidfile.read_text())
    state=Path(f"/proc/{pid}/stat")
    assert not state.exists() or state.read_text().split()[2]=="Z"

def test_timeout_and_cancellation_have_distinct_manifest_states(tmp_path,monkeypatch):
    monkeypatch.setenv("ADAGENT_RUNTIME_ROOT",str(tmp_path))
    @runtime_task("worldengine","timeout")
    def timed():
        run_logged([sys.executable,"-c","import time;time.sleep(60)"],cwd=tmp_path,log_path=current_run("worldengine").logs/"job.log",timeout=0.1)
    with pytest.raises(subprocess.TimeoutExpired):timed()
    manifest=next((tmp_path/"worldengine").glob("*/manifest.json"))
    assert json.loads(manifest.read_text())["status"]=="timed_out"
    @runtime_task("dggt","cancel")
    def cancelled():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):cancelled()
    manifest=next((tmp_path/"dggt").glob("*/manifest.json"))
    assert json.loads(manifest.read_text())["status"]=="cancelled"

def test_trace_links_survive_real_child_process(tmp_path,monkeypatch):
    monkeypatch.setenv("ADAGENT_RUNTIME_ROOT",str(tmp_path))
    @runtime_task("dggt","parent")
    def parent():
        task=current_run("dggt")
        code='from mcp_servers.runtime import runtime_task,current_run\n@runtime_task("dggt","child")\ndef child():\n return {"run_dir":str(current_run("dggt").root)}\nchild()'
        subprocess.run([sys.executable,"-c",code],env=task.child_env(),check=True)
        return str(task.root)
    with trace_context(agent_run_id="agent-123",parent_run_id="agent-123",tool_call_id="call-456"):
        root=Path(parent())
    m=json.loads((root/"manifest.json").read_text())
    assert m["agent_run_id"]==m["parent_run_id"]=="agent-123"
    assert m["tool_call_id"]=="call-456"
