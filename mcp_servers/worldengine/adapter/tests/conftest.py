"""Small preparation boundary stub for orchestration-only filesystem tests."""
import json
from pathlib import Path
import subprocess
import pytest

@pytest.fixture
def stub_training_preparation(monkeypatch):
    import mcp_servers.worldengine.adapter.training as training
    actual = training.run_logged
    def run(command, **kwargs):
        if len(command)>1 and Path(command[1]).name=="prepare_training.py":
            request=json.loads(Path(command[2]).read_text())
            work=Path(request["work_dir"])
            # These tests exercise paths/export isolation, not dataset preparation or GPU updates.
            (work/"training_config.py").write_text(Path(request["config_path"]).read_text())
            (work/"data_preparation.json").write_text(json.dumps({"runtime_root":request["worldengine_root"]}))
            return subprocess.CompletedProcess(command,0,"prepared fixture","")
        return actual(command,**kwargs)
    monkeypatch.setattr(training,"run_logged",run)
