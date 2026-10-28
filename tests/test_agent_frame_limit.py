import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_servers.dggt.adapter.config import ReconstructionConfig
from mcp_servers.dggt.adapter.service import reconstruct as reconstruct_scene
from tools.tool05_instance_discovery.real import run as discover_instances
from tools.tool06_reconstruction.real import run as reconstruct
from tools.tool09_scene_editor.real import run as edit_scene
from tools.tool13_policy_runner.real import run as run_policy


class AgentFrameLimitTests(unittest.TestCase):
    def test_preprocessing_limits_source_to_first_40_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            video.touch()
            config = ReconstructionConfig(input_dir=root / "input", output_dir=root / "output", logs_dir=root / "logs")
            probe = subprocess.CompletedProcess([], 0, json.dumps({"fps": 10, "frames": 198}), "")
            with patch.object(config, "validate"), patch("mcp_servers.dggt.adapter.service.subprocess.run", return_value=probe), patch(
                "mcp_servers.dggt.adapter.service._run_command", side_effect=RuntimeError("stop before GPU execution")
            ) as command:
                with self.assertRaisesRegex(RuntimeError, "stop before GPU execution"):
                    reconstruct_scene(video, config=config)
            args = command.call_args.args[0]
            self.assertEqual(args[args.index("--num-frames") + 1], "40")
            self.assertEqual(float(args[args.index("--duration-seconds") + 1]), 4.0)
            self.assertEqual(config.output_fps, 10)

    def test_dggt_default_is_40_frames(self):
        self.assertEqual(ReconstructionConfig().num_frames, 40)

    def test_reconstruction_forwards_40_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.mp4"
            video.touch()
            completed = subprocess.CompletedProcess([], 0, json.dumps({"status": "success", "result": {}}), "")
            with patch("mcp_servers.dggt.adapter.client.subprocess.run", return_value=completed) as process:
                reconstruct({"video_path": str(video)})
            self.assertEqual(json.loads(process.call_args.kwargs["input"])["num_frames"], 40)

    def test_reconstruction_forwards_native_render_export(self):
        completed = subprocess.CompletedProcess([], 0, json.dumps({"status": "success", "result": {"render_state": "/real/state.pt"}}), "")
        with patch("mcp_servers.dggt.adapter.client.subprocess.run", return_value=completed) as process:
            result = reconstruct({"video_path": __file__, "export_render_state": True})
        self.assertTrue(json.loads(process.call_args.kwargs["input"])["export_render_state"])
        self.assertEqual(result["result"]["render_state"], "/real/state.pt")

    def test_native_export_rejects_instance_edits_before_launch(self):
        with self.assertRaisesRegex(ValueError, "unedited"):
            reconstruct_scene("unused.mp4", edit_config_path="edit.json", export_render_state=True)

    def test_older_64_frame_requests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_frames=40"):
            reconstruct({"num_frames": 64})
        with self.assertRaisesRegex(ValueError, "sequence_length=40"):
            discover_instances({"sequence_length": 64})
        with self.assertRaisesRegex(ValueError, "num_frames=40"):
            edit_scene({"num_frames": 64})
        with self.assertRaisesRegex(ValueError, "max_frame_per_scene=40"):
            run_policy({"max_frame_per_scene": 64})

    def test_policy_runner_forwards_explicit_60_frames(self):
        completed = subprocess.CompletedProcess([], 0, json.dumps({
            "status": "success", "result": {"reports": [[{"succeeded": True}]], "result_dir": "test"},
        }), "")
        with patch("mcp_servers.worldengine.adapter.client.subprocess.run", return_value=completed) as process:
            run_policy({"config_path": "config", "checkpoint_path": "checkpoint",
                        "model_name": "test", "data_type": "worldengine",
                        "max_frame_per_scene": 60})
        self.assertEqual(json.loads(process.call_args.kwargs["input"])["max_frame_per_scene"], 60)

    def test_policy_runner_forwards_40_frames(self):
        completed = subprocess.CompletedProcess([], 0, json.dumps({
            "status": "success", "result": {"reports": [[{"succeeded": True}]], "result_dir": "test"},
        }), "")
        with patch("mcp_servers.worldengine.adapter.client.subprocess.run", return_value=completed) as process:
            run_policy({"config_path": "config", "checkpoint_path": "checkpoint", "model_name": "test", "data_type": "worldengine"})
        self.assertEqual(json.loads(process.call_args.kwargs["input"])["max_frame_per_scene"], 40)


if __name__ == "__main__":
    unittest.main()
