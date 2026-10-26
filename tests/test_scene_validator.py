import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.tool10_scene_validator.real import run


class SceneValidatorTests(unittest.TestCase):
    def test_known_reconstruction_artifacts_do_not_block_edit_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "rendered.mp4"
            video.touch()
            evaluation = {
                "requirement_consistency": True,
                "visual_valid": True,
                "geometry_valid": True,
                "physics_valid": True,
                "traffic_logic_valid": True,
                "issues": [],
                "ignored_reconstruction_artifacts": ["画面中央的黑色横向空洞"],
            }
            with patch("tools.tool10_scene_validator.real._evaluate_qwen", return_value=evaluation):
                result = run({"rendered_video": str(video), "scene_requirement": "前方有救护车"})["result"]
            self.assertTrue(result["passed"])
            self.assertEqual(result["ignored_reconstruction_artifacts"], evaluation["ignored_reconstruction_artifacts"])

    def test_edit_specific_issue_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "rendered.mp4"
            video.touch()
            evaluation = {
                "requirement_consistency": False,
                "visual_valid": False,
                "issues": ["救护车未清晰出现"],
                "ignored_reconstruction_artifacts": ["右侧山体破碎"],
            }
            with patch("tools.tool10_scene_validator.real._evaluate_qwen", return_value=evaluation):
                result = run({"rendered_video": str(video), "scene_requirement": "前方有救护车"})["result"]
            self.assertFalse(result["passed"])
            self.assertEqual(result["issues"], ["救护车未清晰出现"])

    def test_bad_grounding_fails_even_when_requirement_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "rendered.mp4"
            video.touch()
            evaluation = {
                "requirement_consistency": True,
                "visual_valid": True,
                "geometry_valid": False,
                "physics_valid": True,
                "traffic_logic_valid": True,
                "issues": [],
            }
            with patch("tools.tool10_scene_validator.real._evaluate_qwen", return_value=evaluation):
                result = run({"rendered_video": str(video), "scene_requirement": "前车切入"})["result"]
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
