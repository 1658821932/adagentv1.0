import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_servers.ufo.service import (
    REQUESTED_CHECKPOINT, _first_40_frames, _metric_offsets_from_config,
    _verify_model_implementation, reconstruct,
)


class UFOReconstructionTests(unittest.TestCase):
    def setUp(self):
        # These fixtures cover the retained reference-launcher compatibility path.
        runtime = patch.dict(os.environ, {'UFO_RUNTIME': 'legacy'})
        runtime.start()
        self.addCleanup(runtime.stop)

    def test_reference_model_must_match_requested_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            requested = root / "requested" / "ufo"
            reference = root / "reference" / "ufo"
            requested.mkdir(parents=True)
            reference.mkdir(parents=True)
            (requested / "model.py").write_text("version = 1\n")
            (reference / "model.py").write_text("version = 1\n")
            with patch("mcp_servers.ufo.service.REQUESTED_MODEL_SOURCE", requested.parent), patch("mcp_servers.ufo.service.REFERENCE_MODEL_SOURCE", reference.parent):
                _verify_model_implementation()
                (reference / "model.py").write_text("version = 2\n")
                with self.assertRaisesRegex(ValueError, "differs"):
                    _verify_model_implementation()

    def test_verified_native_artifacts_do_not_claim_metric_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            entry = root / "ufo_video_entry_20261012"
            entry.mkdir()
            (entry / "run_video.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            video = root / "input.mp4"
            video.touch()
            workspace = entry / "workspaces" / "test_scene"
            output = workspace / "output" / "worldengine_time_v2"
            output.mkdir(parents=True)
            (workspace / "baselines" / "worldengine_time_v2").mkdir(parents=True)
            (workspace / "baselines" / "worldengine_time_v2" / "READY.json").write_text("{}")
            (output / "DONE.json").write_text(json.dumps({
                "frames": 40,
                "actual_WorldEngine_renderer": True,
                "actual_WorldEngine_asset_manager": True,
            }))
            (output / "reconstruction.mp4").touch()
            asset = root / "WorldEngine" / "data" / "ufo_time_v2" / "assets" / "test_scene"
            asset.mkdir(parents=True)
            (asset / "ASSET.json").write_text(json.dumps({
                "asset_id": "test_scene", "version": "ufo-worldengine-time-v2",
            }))
            (asset / "cameras.npz").touch()
            model = workspace / "reconstruction" / "000" / "pred100k"
            model.mkdir(parents=True)
            (model / "BUILD.json").write_text(json.dumps({
                "checkpoint_step": 99999, "weights_unchanged": True,
                "checkpoint": str(REQUESTED_CHECKPOINT),
            }))
            with patch("mcp_servers.ufo.service._first_40_frames", return_value=video), patch("mcp_servers.ufo.service._verify_model_implementation"), patch("mcp_servers.ufo.service.subprocess.run") as run:
                run.return_value.returncode = 0
                result = reconstruct(str(video), scene_name="test_scene", entry_root=entry)
            self.assertEqual(result["backend"], "ufo_worldengine_time_v2")
            self.assertFalse(result["metric_calibrated"])
            self.assertEqual(result["frame_count"], 40)
            self.assertIn("--baseline", run.call_args.args[0])

    def test_missing_native_artifact_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            entry = root / "entry"
            entry.mkdir()
            (entry / "run_video.sh").touch()
            video = root / "input.mp4"
            video.touch()
            with patch("mcp_servers.ufo.service._first_40_frames", return_value=video), patch("mcp_servers.ufo.service._verify_model_implementation"), patch("mcp_servers.ufo.service.subprocess.run") as run:
                run.return_value.returncode = 0
                with self.assertRaisesRegex(RuntimeError, "required artifacts"):
                    reconstruct(str(video), scene_name="test", entry_root=entry)

    def test_long_input_is_trimmed_to_first_40(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.mp4"
            video.touch()
            with patch("mcp_servers.ufo.service.subprocess.run") as run:
                run.side_effect = [
                    type("Result", (), {"returncode": 0, "stdout": "64\n", "stderr": ""})(),
                    type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
                ]
                output = Path(directory) / "_inputs" / "scene_first40.mp4"
                with patch("pathlib.Path.is_file", return_value=True):
                    result = _first_40_frames(video, Path(directory), "scene")
            self.assertEqual(result, output)
            self.assertIn("-frames:v", run.call_args_list[1].args[0])

    def test_selected_metric_waypoints_become_per_frame_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = root / "edit.json"
            output = root / "offsets.json"
            config.write_text(json.dumps({"Nodes": {"RigidNodes": {"trajectory_edit": {
                "instance_id": [7], "waypoints": [[[0, 0, 0], [0, -3, 0]]],
                "method": ["linear"],
            }}}}), encoding="utf-8")
            _metric_offsets_from_config(config, output, 3, 7)
            values = json.loads(output.read_text())
            self.assertEqual(values["units"], "metres")
            self.assertEqual(values["offsets"], [[0.0, 0.0, 0.0], [0.0, -1.5, 0.0], [0.0, -3.0, 0.0]])
            with self.assertRaisesRegex(ValueError, "changed the selected target"):
                _metric_offsets_from_config(config, output, 3, 8)


if __name__ == "__main__":
    unittest.main()
