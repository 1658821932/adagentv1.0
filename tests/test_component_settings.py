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
    logger=RunLogger("test",3,"test","https://example.invalid",{},tmp_path,llm_reasoning_effort="high")
    logger.start_tool_call(0,"policy_runner",{"video_path":"/video"})
    logger.save_state({"status":"running","current_step":0,"history":[{"tool":"policy_runner","result":{"status":"success"}}]})
    assert json.loads((logger.run_dir/"run.json").read_text())["llm"]["reasoning_effort"]=="high"
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


def test_editing_commands_resolve_bundled_paths_outside_project_cwd(tmp_path, monkeypatch):
    from mcp_servers.scene_editing.adapter import config
    project = tmp_path / "project"
    component = project / "mcp_servers/scene_editing"
    (component / "adapter").mkdir(parents=True)
    interpreter = project / "environments/scene_editing/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to("/usr/bin/python3")
    command = ["environments/scene_editing/bin/python",
               "mcp_servers/scene_editing/core/adversarial/editing_backend.py",
               "--fixture-argument"]
    absolute = ["/fixture/python", "/fixture/backend.py"]
    (component / "adapter/settings.json").write_text(json.dumps({"commands": {
        "attack_editing": command, "ego_editing": absolute,
    }}))
    monkeypatch.setattr(config, "COMPONENT_ROOT", component)
    monkeypatch.setattr(config, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    assert config.stage_command("attack_editing") == [
        str(interpreter), str(project / command[1]), "--fixture-argument",
    ]
    assert config.stage_command("ego_editing") == absolute
    assert config.stage_command("missing") is None
