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




def test_wfa_reads_only_this_task_output(tmp_path):
    from mcp_servers.worldengine_failure_analysis.adapter.service import analyze_scene
    config = tmp_path / "config.json"; config.write_text("{}")
    def backend(command, **kwargs):
        output = Path(command[command.index("--run-dir") + 1])
        scene = output / "scenes/token"; scene.mkdir(parents=True)
        (scene / "result.json").write_text(json.dumps({"token": "token", "status": "supported", "failure_reason": "test"}))
        (scene / "status.json").write_text('{"state":"completed"}')
        return subprocess.CompletedProcess(command, 0, "done", "")
    with patch("mcp_servers.worldengine_failure_analysis.adapter.service._check_secret"), patch("mcp_servers.worldengine_failure_analysis.adapter.service.run_logged", side_effect=backend):
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
