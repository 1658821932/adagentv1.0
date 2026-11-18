"""Tool 01 must classify the other vehicle's origin from ego-relative headings only.

The archive stores every box in the ego frame of its own frame: position in metres
(x forward, y left) and yaw as a heading already relative to the ego heading
(WorldEngine/projects/SimEngine/worldengine/manager/data_manager.py:200-220).
Classifying an origin therefore must not involve the ego world heading again.
"""
import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.tool01_failure_analyzer import structure_facts as facts  # noqa: E402

STEP_S = 0.5
FRAME_COUNT = 12


def ego_matrix(x, y, yaw):
    cos, sin = math.cos(yaw), math.sin(yaw)
    return np.array([[cos, -sin, 0.0, x],
                     [sin, cos, 0.0, y],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def ego_relative(ego_pose, world_pose):
    """World pose -> archive box convention: metres in the ego frame, yaw relative to ego."""
    ego_x, ego_y, ego_yaw = ego_pose
    world_x, world_y, world_yaw = world_pose
    dx, dy = world_x - ego_x, world_y - ego_y
    cos, sin = math.cos(-ego_yaw), math.sin(-ego_yaw)
    return (dx * cos - dy * sin, dx * sin + dy * cos, world_yaw - ego_yaw)


def compose(ego_pose, box):
    """Place an ego-frame box back into the world."""
    ego_x, ego_y, ego_yaw = ego_pose
    x, y, yaw = box
    cos, sin = math.cos(ego_yaw), math.sin(ego_yaw)
    return (ego_x + x * cos - y * sin, ego_y + x * sin + y * cos, ego_yaw + yaw)


def build_frames(boxes, ego_pose):
    """Synthetic archive frames.

    boxes: ego-frame boxes (x, y, yaw) as stored by the simulator.
    ego_pose: (x, y, yaw) in the world, used only for ego2global.
    """
    frames = []
    for index in range(FRAME_COUNT):
        rows = [[x, y, 0.0, 4.6, 1.9, 1.6, yaw] for x, y, yaw in boxes]
        frames.append({
            "timestamp": 1_600_000_000_000_000 + index * int(STEP_S * 1e6),
            "ego2global": ego_matrix(*ego_pose),
            "map_location": "synthetic",
            "cams": {},
            "anns": {
                "track_tokens": [f"track{position}" for position in range(len(rows))],
                "gt_names": ["vehicle"] * len(rows),
                "gt_boxes": np.array(rows, dtype=float).reshape(-1, 7),
                "gt_velocity_3d": np.zeros((len(rows), 3), dtype=float),
            },
        })
    return frames


def frames_from_world(ego_pose, world_poses):
    """Same archive frames, entered through world coordinates the way the simulator does."""
    boxes = [ego_relative(ego_pose, pose) for pose in world_poses]
    return build_frames(boxes, ego_pose)


def origin_from_frames(frames):
    """Origin of the tracked vehicle, measured from its first observed frame."""
    sequence = next(iter(facts.actor_sequences(frames).values()))
    return facts.classify_origin(sequence[0])


# (expected origin, ego-frame pose of the other vehicle)
CASES = (
    ("ahead_same_lane", (15.0, 0.0, 0.0)),
    ("behind_same_lane", (-12.0, 0.0, 0.0)),
    ("oncoming", (15.0, 0.0, math.pi)),
    ("left_adjacent", (-2.0, 5.0, 0.0)),
    ("right_adjacent", (-2.0, -5.0, 0.0)),
    ("left_arm", (-5.0, 20.0, math.radians(80.0))),
    ("right_arm", (5.0, -20.0, math.radians(80.0))),
    ("right_arm", (5.0, -20.0, math.pi)),
    ("left_arm", (-5.0, 20.0, -math.radians(120.0))),
    ("left_arm", (-20.0, 40.0, 0.0)),
)


class OriginTests(unittest.TestCase):
    def test_signature_takes_no_ego_heading(self):
        self.assertEqual(list(inspect.signature(facts.classify_origin).parameters), ["first"])

    def test_origin_matches_the_relative_geometry(self):
        for expected, box in CASES:
            with self.subTest(expected=expected, box=box):
                self.assertEqual(origin_from_frames(build_frames([box], (0.0, 0.0, 0.0))), expected)

    def test_ego_world_heading_does_not_change_origin(self):
        headings = (0.0, math.radians(90.0), math.pi, math.radians(152.8), math.radians(-127.3))
        for expected, box in CASES:
            for heading in headings:
                with self.subTest(expected=expected, ego_heading=round(math.degrees(heading), 1)):
                    frames = build_frames([box], (12.0, -30.0, heading))
                    self.assertEqual(origin_from_frames(frames), expected)

    def test_whole_scene_rotation_keeps_origin(self):
        angles = (0.0, math.radians(37.0), math.radians(90.0), math.radians(152.8),
                  math.radians(-127.3), math.pi)
        ego = (12.0, -30.0, math.radians(64.0))
        for expected, box in CASES:
            actor = compose(ego, box)
            self.assertEqual(ego_relative(ego, actor)[2], box[2])  # compose/reduce agree
            for angle in angles:
                cos, sin = math.cos(angle), math.sin(angle)
                rotate = lambda pose: (pose[0] * cos - pose[1] * sin,
                                       pose[0] * sin + pose[1] * cos,
                                       pose[2] + angle)
                frames = frames_from_world(rotate(ego), [rotate(actor)])
                with self.subTest(expected=expected, angle=round(math.degrees(angle), 1)):
                    self.assertEqual(origin_from_frames(frames), expected)

    def test_origin_describes_the_entry_not_the_end(self):
        frames = build_frames([(4.0, -20.0, math.radians(80.0))], (0.0, 0.0, 0.0))
        for index, frame in enumerate(frames):
            if index >= FRAME_COUNT // 2:  # drives up and merges ahead of the ego
                frame["anns"]["gt_boxes"][0][0] = 4.0 + index * 2.0
                frame["anns"]["gt_boxes"][0][1] = -20.0 + index * 3.0
                frame["anns"]["gt_boxes"][0][6] = 0.0
        self.assertEqual(origin_from_frames(frames), "right_arm")


if __name__ == "__main__":
    unittest.main()