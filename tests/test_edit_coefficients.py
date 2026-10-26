import json
import tempfile
import unittest
from pathlib import Path

from tools.tool09_scene_editor.coordinates import convert_edit_config


class EditCoefficientTests(unittest.TestCase):
    def test_trajectory_requires_explicit_conversion_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            camera_path = Path(directory) / "camera_params.json"
            camera_path.write_text(json.dumps({"frames": [{
                "frame_index": 0,
                "world_to_camera_opencv": [
                    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                ],
            }]}), encoding="utf-8")
            config = {"Nodes": {"RigidNodes": {"trajectory_edit": {
                "instance_id": [0], "waypoints": [[[1, 0, 0]]], "method": ["linear"],
            }}}}
            with self.assertRaisesRegex(ValueError, "explicit meters_to_dggt"):
                convert_edit_config(config, camera_path, None)
            converted, _ = convert_edit_config(config, camera_path, 0.02)
            self.assertEqual(
                converted["Nodes"]["RigidNodes"]["trajectory_edit"]["waypoints"],
                [[[0.0, 0.0, 0.02]]],
            )

    def test_trajectory_uses_each_frames_camera_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            camera_path = Path(directory) / "camera_params.json"
            camera_path.write_text(json.dumps({"frames": [
                {"frame_index": 0, "world_to_camera_opencv": [
                    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
                {"frame_index": 1, "world_to_camera_opencv": [
                    [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]},
            ]}), encoding="utf-8")
            config = {"Nodes": {"RigidNodes": {"trajectory_edit": {
                "instance_id": [0], "waypoints": [[[1, 0, 0]]], "method": ["linear"],
            }}}}
            converted, diagnostics = convert_edit_config(config, camera_path, 0.01)
            self.assertEqual(converted["Nodes"]["RigidNodes"]["trajectory_edit"]["waypoints"],
                             [[[0.0, 0.0, 0.01], [-0.01, 0.0, 0.0]]])
            self.assertEqual(diagnostics["conversion_mode"], "per_frame_camera_pose")

    def test_local_waypoints_are_interpolated_before_world_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            camera_path = Path(directory) / "camera_params.json"
            frames = [{"frame_index": index, "world_to_camera_opencv": [
                [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
                for index in range(3)]
            camera_path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
            config = {"Nodes": {"RigidNodes": {"trajectory_edit": {
                "instance_id": [0], "waypoints": [[[0, 0, 0], [0, -2, 0]]], "method": ["linear"],
            }}}}
            converted, _ = convert_edit_config(config, camera_path, 0.01)
            self.assertEqual(converted["Nodes"]["RigidNodes"]["trajectory_edit"]["waypoints"],
                             [[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]]])

    def test_camera_pitch_cannot_move_vehicle_vertically(self):
        with tempfile.TemporaryDirectory() as directory:
            camera_path = Path(directory) / "camera_params.json"
            camera_path.write_text(json.dumps({"frames": [{
                "frame_index": 0,
                "world_to_camera_opencv": [
                    [1, 0, 0, 0], [0, 0.8, 0.6, 0], [0, -0.6, 0.8, 0], [0, 0, 0, 1],
                ],
            }]}), encoding="utf-8")
            config = {"Nodes": {"RigidNodes": {"trajectory_edit": {
                "instance_id": [0], "waypoints": [[[10, -2, 0]]], "method": ["linear"],
            }}}}
            converted, diagnostics = convert_edit_config(config, camera_path, 0.01)
            self.assertEqual(
                converted["Nodes"]["RigidNodes"]["trajectory_edit"]["waypoints"],
                [[[0.02, 0.0, 0.08]]],
            )
            self.assertEqual(diagnostics["vertical_axis_locked"], "world_y")
