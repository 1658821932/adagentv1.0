"""Task isolation and adapter compatibility without loading GPU models."""
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_servers.runtime import current_run, runtime_task




def test_wfa_reads_only_this_task_output(tmp_path, monkeypatch):
    monkeypatch.setenv("WFA_PYTHON", sys.executable)
    monkeypatch.setenv("ADAGENT_RUNTIME_ROOT", str(tmp_path/'runtime'))
    monkeypatch.setenv('WFA_LIBRARY_DIR',str(tmp_path))
    monkeypatch.setenv('LD_LIBRARY_PATH','/fixture/existing-libs')
    monkeypatch.setenv('ADAGENT_AGENT_RUN_ID','fixture-main')
    monkeypatch.setattr('common.api_profiles.profile_argument',lambda component:'fixture-model-profile')
    from mcp_servers.worldengine_failure_analysis.adapter.service import analyze_scene
    config = tmp_path / "config.json"; config.write_text("{}")
    def backend(command, **kwargs):
        assert kwargs['env']['LD_LIBRARY_PATH']==str(tmp_path)+os.pathsep+'/fixture/existing-libs'
        assert kwargs['env']['WFA_LLM_PROFILE']=='fixture-model-profile'
        assert kwargs['env']['ADAGENT_AGENT_RUN_ID']=='fixture-main'
        output = Path(command[command.index("--run-dir") + 1])
        scene = output / "scenes/token"; scene.mkdir(parents=True)
        payload={"token": "token", "status": "supported", "failure_reason": "test"}
        # Native WFA returns RESULT on stdout, captured by run_logged.
        Path(kwargs['log_path']).write_text('RESULT '+json.dumps(payload)+'\n')
        (scene / "status.json").write_text('{"state":"completed"}')
        return subprocess.CompletedProcess(command, 0, "done", "")
    with patch("mcp_servers.worldengine_failure_analysis.adapter.service.run_logged", side_effect=backend):
        first = analyze_scene("token", config_path=str(config))
        second = analyze_scene("token", config_path=str(config))
    assert first.run_dir != second.run_dir
    assert Path(first.result_path).is_relative_to(Path(first.run_dir) / "output")

def test_wfa_report_cli_accepts_existing_task_output(tmp_path):
    import csv
    archive = tmp_path / "archive"
    metrics = archive / "WE_output/openscene_format"
    metrics.mkdir(parents=True)
    with (archive / "scene_index.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["token"])
        writer.writeheader(); writer.writerow({"token": "token"})
    with (metrics / "all_scenes_pdm_averages_R.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["token", "no_at_fault_collisions", "drivable_area_compliance"])
        writer.writeheader(); writer.writerow({"token": "token", "no_at_fault_collisions": 1, "drivable_area_compliance": 0})
    cfg = tmp_path / "cfg.json"; cfg.write_text(json.dumps({"result_root": str(archive)}))
    output = tmp_path / "existing-task/output"
    output.mkdir(parents=True)
    subprocess.run([sys.executable, "mcp_servers/worldengine_failure_analysis/core/run.py", "report", "--config", str(cfg), "--run-dir", str(output)], check=True, capture_output=True, text=True)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["official_failure_count"] == 1
    assert (output / "report.md").is_file()


def test_wfa_library_path_preserves_existing_env_and_trace(tmp_path, monkeypatch):
    from mcp_servers.worldengine_failure_analysis.adapter.config import runtime_env
    monkeypatch.setenv('WFA_LIBRARY_DIR',str(tmp_path))
    source=dict(LD_LIBRARY_PATH='/existing/lib',ADAGENT_AGENT_LOG_DIR='/fixture/run')
    env=runtime_env(source)
    assert env['LD_LIBRARY_PATH']==str(tmp_path)+os.pathsep+'/existing/lib'
    assert source['LD_LIBRARY_PATH']=='/existing/lib'
    assert env['ADAGENT_AGENT_LOG_DIR']=='/fixture/run'
    assert runtime_env(env)==env


def test_wfa_dependency_failure_preserves_cause_and_is_not_an_archive_retry(tmp_path, monkeypatch):
    from mcp_servers.worldengine_failure_analysis.adapter import service
    from tools.tool01_failure_analyzer.real import run
    from agent.state import AgentState
    monkeypatch.setenv('WFA_PYTHON',sys.executable)
    monkeypatch.setenv('ADAGENT_RUNTIME_ROOT',str(tmp_path/'runtime'))
    config=tmp_path/'config.json';config.write_text('{}')
    monkeypatch.setattr(service,'DEFAULT_CONFIG',config)
    archive=tmp_path/'archive';archive.mkdir()
    (archive/'scene_index.csv').write_text('token\nfull-token\n')
    monkeypatch.setattr(service,'resolve_simulation',lambda spec:archive)
    monkeypatch.setattr(service,'_config_for_simulation',lambda *args:config)
    def backend(command, **kwargs):
        output=Path(command[command.index('--run-dir')+1])
        scene=output/'scenes/full-token';scene.mkdir(parents=True)
        (scene/'status.json').write_text(json.dumps(dict(state='failed',error_type='ImportError',
            message='libGL.so.1: cannot open shared object file: No such file or directory')))
        return subprocess.CompletedProcess(command,2,'','')
    monkeypatch.setattr(service,'run_logged',backend)
    result=run(dict(token='token',simulation='fixture/archive'))
    assert result['status']=='error'
    error=result['error']
    assert error['type']=='backend_environment_error' and error['retryable'] is False
    assert error['failure_stage']=='environment' and 'libGL.so.1' in error['message']
    assert error['archive_validated'] and error['scene_validated']
    reference=tmp_path/'tool-result.json';reference.write_text(json.dumps(result))
    state=AgentState('fixture');state.add_tool_result('failure_analyzer',result,result_ref=str(reference))
    compact=state.to_orchestrator_dict()
    assert 'backend_environment_error' in json.dumps(compact)
    assert 'archive_validated' in json.dumps(compact)


@pytest.mark.parametrize('details',[None,'missing','not-a-dict'])
def test_wfa_error_wrapper_handles_old_exception_without_hiding_backend_cause(monkeypatch,details):
    from tools.tool01_failure_analyzer import real
    error=real.WfaError('fixture underlying backend failure')
    if details=='missing':
        del error.details
    else:
        error.details=details
    def backend(*args,**kwargs):
        raise error
    monkeypatch.setattr(real,'analyze_scene',backend)
    result=real.run(dict(token='fixture'))
    assert result['status']=='error'
    assert result['error']['type']=='failure_analysis_error'
    assert result['error']['message']=='fixture underlying backend failure'
