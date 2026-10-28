import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.tool09_scene_editor.real import run


class SceneEditorEgoOffsetsTest(unittest.TestCase):
    def test_ego_offsets_only_are_forwarded_to_dggt(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.mp4"
            offsets = Path(directory) / "offsets.json"
            video.touch()
            offsets.write_text("{}", encoding="utf-8")
            response = subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=json.dumps({"status": "success", "result": {"rendered_video": "rendered.mp4"}}),
                stderr="",
            )
            with patch("mcp_servers.dggt.adapter.editing.subprocess.run", return_value=response) as command:
                result = run({"video_path": str(video), "ego_trajectory_offsets_path": str(offsets)})
            payload = json.loads(command.call_args.kwargs["input"])
            self.assertIsNone(payload["edit_config_path"])
            self.assertEqual(payload["ego_trajectory_offsets_path"], str(offsets))
            self.assertEqual(payload["num_frames"], 40)
            self.assertIsNone(result["result"]["scene_config"])

    def test_requires_an_edit_or_ego_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "input.mp4"
            video.touch()
            with self.assertRaisesRegex(ValueError, "edit_config_path or ego_trajectory_offsets_path"):
                run({"video_path": str(video)})
