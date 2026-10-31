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

def test_smartedit_llm_settings_do_not_read_agent_json(tmp_path,monkeypatch):
    import mcp_servers.smart_edit.adapter.config as config
    monkeypatch.setattr(config,"PROJECT_ROOT",tmp_path/"no-agent-config")
    with patch("dotenv.load_dotenv"),patch.dict("os.environ",{"OPENAI_API_KEY":"test"},clear=True),patch.object(config,"settings",return_value={"llm":{"model":"editor-model","base_url":"https://editor.invalid/v1","timeout":123,"max_tokens":321}}):
        values=config.llm_settings()
    assert values["model"]=="editor-model"
    assert values["timeout"]==123 and values["max_tokens"]==321

def test_visual_exclusions_are_per_scene_not_global(tmp_path,monkeypatch):
    import tools.tool10_scene_validator.real as validator
    monkeypatch.setenv("OPENAI_API_KEY","test")
    frame=tmp_path/"frame.jpg";frame.write_bytes(b"test")
    received=[]
    def chat(url,key,payload,timeout):
        received.append(payload["messages"][0]["content"][0]["text"])
        return {"choices":[{"message":{"content":json.dumps({"issues":[]})}}]}
    with patch.object(validator,"_dashscope_chat",side_effect=chat),patch.object(validator,"_image_data_url",return_value="data:image/jpeg;base64,dGVzdA=="):
        validator._evaluate_qwen({"frame_paths":[str(frame)]},Path("/video"),"lead car")
        validator._evaluate_qwen({"frame_paths":[str(frame)],"ignored_reconstruction_artifacts":["known source-only artifact"]},Path("/video"),"lead car")
    assert "右侧山体" not in received[0] and "黑色横向空洞" not in received[0]
    assert "known source-only artifact" not in received[0]
    assert "known source-only artifact" in received[1]

def test_agent_snapshot_is_written_after_completed_tool(tmp_path):
    from agent.logger import RunLogger
    logger=RunLogger("test",3,"test","https://example.invalid",{},logs_root=tmp_path)
    logger.start_tool_call(0,"reconstruction",{"video_path":"/video"})
    logger.save_state({"status":"running","current_step":0,"history":[{"tool":"reconstruction","result":{"status":"success"}}]})
    assert json.loads((logger.run_dir/"state.json").read_text())["history"][0]["tool"]=="reconstruction"
    assert json.loads((logger.tools_dir/"step_00_reconstruction/call.json").read_text())["tool_call_id"].endswith(":step_00:reconstruction")

def test_interpreter_symlink_keeps_virtual_environment(tmp_path):
    import subprocess,sys
    env=tmp_path/"isolated-venv"
    subprocess.run([sys.executable,"-m","venv","--without-pip",str(env)],check=True)
    python=runtime_path(tmp_path,"python","ABSENT_INTERPRETER_OVERRIDE",str(env/"bin/python"))
    assert str(python)==str(env/"bin/python")
    prefix=subprocess.check_output([str(python),"-c","import sys;print(sys.prefix)"],text=True).strip()
    assert Path(prefix)==env
