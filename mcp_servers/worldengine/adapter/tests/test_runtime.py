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




def test_simulation_directs_backend_into_distinct_output_dirs(tmp_path):
    from mcp_servers.worldengine.adapter.config import WorldEngineConfig
    from mcp_servers.worldengine.adapter.service import run_simulation
    script = tmp_path / "simulate.sh"
    script.write_text('mkdir -p "$WORLDENGINE_TEST_PATH"\nprintf \'[{"succeeded": true}]\' > "$WORLDENGINE_TEST_PATH/runner_report_0.json"\n')
    config_file, checkpoint = tmp_path / "cfg.py", tmp_path / "model.pth"
    config_file.touch(); checkpoint.touch()
    config = WorldEngineConfig(project_root=tmp_path, script=script)
    results = [run_simulation(str(config_file), str(checkpoint), "same", "same", config=config) for _ in range(2)]
    assert results[0]["result_dir"] != results[1]["result_dir"]
    for result in results:
        root = Path(result["run_dir"])
        assert Path(result["result_dir"]).is_relative_to(root / "output")
        assert Path(result["log_path"]).is_relative_to(root / "logs")
        assert result["reports"] == [[{"succeeded": True}]]
    assert not (tmp_path / "experiments/closed_loop_exps").exists()

def test_training_ignores_stale_checkpoints_and_keeps_export(tmp_path):
    from mcp_servers.worldengine.adapter.training import train_policy
    core = tmp_path / "core"
    script = core / "projects/AlgEngine/scripts/train.py"
    script.parent.mkdir(parents=True)
    script.write_text('import sys\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index("--work-dir")+1]);(p/"epoch_1.pth").write_bytes(b"fresh")\nprint("loss.imi: 0.25")\n')
    files = {}
    for name in ("ann", "tokens", "base", "config"):
        files[name] = tmp_path / name
        files[name].touch()
    export = tmp_path / "exports"; export.mkdir()
    (export / "epoch_99.pth").write_bytes(b"stale")
    result = train_policy({
        "training_data": {"ann_file": str(files["ann"]), "token_file": str(files["tokens"])},
        "base_checkpoint": str(files["base"]), "config_path": str(files["config"]),
        "worldengine_root": str(core), "python": sys.executable, "work_dir": str(export),
    })
    assert Path(result["checkpoint"]).read_bytes() == b"fresh"
    assert Path(result["checkpoint"]).is_relative_to(Path(result["run_dir"]) / "output")
    assert result["final_loss"] == 0.25
    assert (Path(result["export_dir"]) / "epoch_1.pth").read_bytes() == b"fresh"
    assert (export / "epoch_99.pth").read_bytes() == b"stale"

def test_worldengine_evaluation_stores_result_with_executed_trajectory(tmp_path):
    import pickle
    from mcp_servers.worldengine.adapter.evaluation import filter_failure_cases
    scene = tmp_path / "scene.pkl"
    state = {"position": [[0, 0, 0]], "length": [4], "width": [2], "valid": [True]}
    scene.write_bytes(pickle.dumps([{"object_track": {"ego": {"state": state}, "other": {"state": state}}}]))
    result = filter_failure_cases({"scenario_pkl": str(scene), "python": sys.executable})
    assert result["is_failure_case"] is True
    root = Path(result["run_dir"])
    assert (root / "output/result.json").is_file()
    assert (root / "logs/evaluation.log").is_file()
