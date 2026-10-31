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




def test_smart_edit_internal_output_and_explicit_export(tmp_path):
    from mcp_servers.smart_edit.adapter.service import generate_edit_config
    requested = tmp_path / "export/config.json"
    def backend(command, **kwargs):
        payload = json.loads(kwargs["input"])
        path = Path(payload["output_path"])
        path.write_text(json.dumps({"coordinate_system": "camera_local_forward_left_up", "offsets": [[0, 0, 0]]}))
        return subprocess.CompletedProcess(command, 0, json.dumps({"status": "success", "result": {"saved_path": str(path)}}), "")
    with patch("mcp_servers.smart_edit.adapter.service.run_logged", side_effect=backend):
        result = generate_edit_config({"edit_plan": {"request": "stay"}, "output_path": str(requested), "output_kind": "ego_trajectory_offsets"})
    assert requested.read_text() == Path(result["saved_path"]).read_text()
    assert Path(result["saved_path"]).is_relative_to(Path(result["run_dir"]) / "output")
