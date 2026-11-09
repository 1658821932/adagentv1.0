"""Service settings do not depend on Agent orchestration settings."""
import json
from pathlib import Path
from unittest.mock import patch
from mcp_servers.configuration import runtime_path

def test_settings_path_and_environment_override(tmp_path,monkeypatch):
    component=tmp_path/"service";(component/"adapter").mkdir(parents=True)
    (component/"adapter/settings.json").write_text(json.dumps({"paths":{"python":str(tmp_path/"configured-python")}}))
    assert runtime_path(component,"python","TEST_SERVICE_PYTHON","unused")==tmp_path/"configured-python"
    monkeypatch.setenv("TEST_SERVICE_PYTHON",str(tmp_path/"override-python"))
    assert runtime_path(component,"python","TEST_SERVICE_PYTHON","unused")==tmp_path/"override-python"



def test_agent_snapshot_is_written_after_completed_tool(tmp_path):
    from agent.logger import RunLogger
    logger=RunLogger("test",3,"test","https://example.invalid",{},logs_root=tmp_path)
    logger.start_tool_call(0,"policy_runner",{"video_path":"/video"})
    logger.save_state({"status":"running","current_step":0,"history":[{"tool":"policy_runner","result":{"status":"success"}}]})
    assert json.loads((logger.run_dir/"state.json").read_text())["history"][0]["tool"]=="policy_runner"
    assert json.loads((logger.tools_dir/"step_00_policy_runner/call.json").read_text())["tool_call_id"].endswith(":step_00:policy_runner")

def test_interpreter_symlink_keeps_virtual_environment(tmp_path):
    import subprocess,sys
    env=tmp_path/"isolated-venv"
    subprocess.run([sys.executable,"-m","venv","--without-pip",str(env)],check=True)
    python=runtime_path(tmp_path,"python","ABSENT_INTERPRETER_OVERRIDE",str(env/"bin/python"))
    assert str(python)==str(env/"bin/python")
    prefix=subprocess.check_output([str(python),"-c","import sys;print(sys.prefix)"],text=True).strip()
    assert Path(prefix)==env
