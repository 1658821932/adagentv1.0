import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_dggt_instance_for_worldengine import _camera_for_worldengine


class PrepareDggtWorldEngineTests(unittest.TestCase):
    def test_rendered_ego_pose_is_normalized_for_worldengine(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            ply = {
                "coordinate_convention": "OpenCV world-to-camera for normalized PLY coordinates",
                "ply_world_transform": {"center": [10, 20, 30], "scale": 10},
                "frames": [{"frame_index": 0, "intrinsics": [[1]], "world_to_camera_opencv": identity}],
            }
            rendered = {
                "coordinate_convention": "OpenCV world-to-camera (x right, y down, z forward)",
                "frames": [{"frame_index": 0, "intrinsics": [[2]], "world_to_camera_opencv": identity}],
            }
            (scene / "ply_camera_params.json").write_text(json.dumps(ply), encoding="utf-8")
            (scene / "camera_params.json").write_text(json.dumps(rendered), encoding="utf-8")
            result, source = _camera_for_worldengine(scene, 1)
            self.assertEqual(source, "camera_params.json")
            self.assertEqual(result["frames"][0]["world_to_camera_opencv"][0][3], 1.0)
            self.assertTrue(result["frames"][0]["world_to_camera_opencv"][1][3] == 2.0)
            self.assertTrue(result["frames"][0]["world_to_camera_opencv"][2][3] == 3.0)


if __name__ == "__main__":
    unittest.main()
