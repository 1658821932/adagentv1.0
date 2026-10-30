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




def test_dggt_reconstruction_artifacts_are_all_under_one_task(tmp_path):
    from mcp_servers.dggt.adapter.config import ReconstructionConfig
    from mcp_servers.dggt.adapter.service import reconstruct
    video = tmp_path / "video.mp4"; video.write_bytes(b"source")
    def command(args, **kwargs):
        if "--output_path" in args:
            output = Path(args[args.index("--output_path") + 1])
            clouds = output / "point_clouds"; clouds.mkdir()
            for name in ("combined", "static", "sky"):
                (clouds / (name + "_001.ply")).touch()
            scene = output / "001"; scene.mkdir()
            for name in ("ply_camera_params.json", "camera_params.json", "supersplat_camera_poses.json", "rendered_video.mp4"):
                (scene / name).touch()
        kwargs["log_path"].write_text("completed")
    with patch.object(ReconstructionConfig, "validate"), patch("mcp_servers.dggt.adapter.service.subprocess.run", return_value=subprocess.CompletedProcess([], 0, '{"fps":8,"frames":40}', "")), patch("mcp_servers.dggt.adapter.service._run_command", side_effect=command):
        result = reconstruct(str(video))
    root = Path(result.run_dir)
    assert Path(result.input_dir) == root / "input"
    assert Path(result.output_dir) == root / "output"
    assert Path(result.scene_ply).is_relative_to(root / "output")
    assert (root / "logs/01_preprocess.log").is_file()
    assert (root / "logs/02_reconstruction.log").is_file()
    assert video.read_bytes() == b"source"

def test_instance_discovery_keeps_explicit_export(tmp_path):
    from mcp_servers.dggt.adapter.instances import discover_instances
    image_dir = tmp_path / "images"; image_dir.mkdir()
    exported = tmp_path / "export/instances.json"
    def backend(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"instances":[]}')
        return subprocess.CompletedProcess(command, 0, "done", "")
    with patch("mcp_servers.dggt.adapter.instances.subprocess.run", side_effect=backend):
        result = discover_instances({"image_dir": str(image_dir), "output_path": str(exported)})
    assert exported.read_text() == Path(result["instance_info_path"]).read_text()
    assert Path(result["instance_info_path"]).is_relative_to(Path(result["run_dir"]) / "output")

def test_real_child_process_joins_parent_workspace():
    @runtime_task("dggt", "parent")
    def parent():
        task = current_run("dggt")
        code = ('from mcp_servers.runtime import runtime_task,current_run\n'
                '@runtime_task("dggt","child")\n'
                'def child():\n'
                ' r=current_run("dggt");(r.output/"child.txt").write_text(str(r.root))\n'
                'child()\n')
        subprocess.run([sys.executable, "-c", code], env=task.child_env(), check=True)
        return {"run_dir": str(task.root)}
    result = parent()
    root = Path(result["run_dir"])
    assert (root / "output/child.txt").read_text() == str(root)
    assert json.loads((root / "manifest.json").read_text())["operation"] == "parent"
