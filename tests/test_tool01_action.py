"""Tool 01 must measure the other vehicle's own turn, not ego yaw minus actor yaw.

The archive stores local yaw = actor heading - ego heading of the same frame
(data_manager.py:200-220), so the actor's world heading has to be rebuilt per
observation before any turn angle is measured.
"""
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.tool01_failure_analyzer import structure_facts as facts  # noqa: E402
from tools.tool01_failure_analyzer.lane_behavior import direction_action  # noqa: E402

N, DT, SPEED = 12, 0.5, 8.0


def normalized(angle):
    """The archive stores every heading folded into [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def ego_matrix(x, y, yaw):
    cos, sin = math.cos(yaw), math.sin(yaw)
    return np.array([[cos, -sin, 0.0, x],
                     [sin, cos, 0.0, y],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def to_world(ego, box):
    ego_x, ego_y, ego_yaw = ego
    x, y, yaw = box
    cos, sin = math.cos(ego_yaw), math.sin(ego_yaw)
    return (ego_x + x * cos - y * sin, ego_y + x * sin + y * cos, ego_yaw + yaw)


def to_box(ego, world):
    ego_x, ego_y, ego_yaw = ego
    x, y, yaw = world
    dx, dy = x - ego_x, y - ego_y
    cos, sin = math.cos(-ego_yaw), math.sin(-ego_yaw)
    return (dx * cos - dy * sin, dx * sin + dy * cos, yaw - ego_yaw)


def scene(ego_yaw0, ego_turn, actor_yaw0, actor_turn, start=(12.0, 0.0, 0.0),
          present=None, speed=SPEED):
    """Per-frame ego pose, actor driving at constant speed along its own heading."""
    pose = to_world((0.0, 0.0, ego_yaw0), start)
    frames = []
    for index in range(N):
        fraction = index / (N - 1)
        ego = (0.0, 0.0, ego_yaw0 + ego_turn * fraction)
        actor_yaw = actor_yaw0 + actor_turn * fraction
        rows = []
        if present is None or index in present:
            x, y, local_yaw = to_box(ego, (pose[0], pose[1], actor_yaw))
            rows.append([x, y, 0.0, 4.6, 1.9, 1.6, normalized(local_yaw)])
        frames.append({
            "timestamp": 1_600_000_000_000_000 + index * int(DT * 1e6),
            "ego2global": ego_matrix(*ego),
            "map_location": "synthetic",
            "cams": {},
            "anns": {
                "track_tokens": ["track"] * len(rows),
                "gt_names": ["vehicle"] * len(rows),
                "gt_boxes": np.array(rows, dtype=float).reshape(-1, 7),
                "gt_velocity_3d": np.array([[speed, 0.0, 0.0]] * len(rows), dtype=float).reshape(-1, 3),
            },
        })
        pose = (pose[0] + speed * DT * math.cos(actor_yaw),
                pose[1] + speed * DT * math.sin(actor_yaw), actor_yaw)
    return frames


def frames_with_headings(ego_yaw_at, actor_yaw_at, offset=(12.0, 0.0)):
    """Frames whose per-frame ego and actor headings are given explicitly."""
    frames = []
    for index in range(N):
        ego_yaw = ego_yaw_at(index)
        local_yaw = normalized(actor_yaw_at(index) - ego_yaw)
        frames.append({
            "timestamp": 1_600_000_000_000_000 + index * int(DT * 1e6),
            "ego2global": ego_matrix(0.0, 0.0, ego_yaw),
            "map_location": "synthetic",
            "cams": {},
            "anns": {
                "track_tokens": ["track"],
                "gt_names": ["vehicle"],
                "gt_boxes": np.array([[offset[0], offset[1], 0.0, 4.6, 1.9, 1.6, local_yaw]], dtype=float),
                "gt_velocity_3d": np.array([[SPEED, 0.0, 0.0]], dtype=float),
            },
        })
    return frames


def net_world_turn(sequence):
    """Numeric net heading change the tool measures, in degrees."""
    unwrapped = [sequence[0]["world_yaw"]]
    for item in sequence[1:]:
        delta = item["world_yaw"] - unwrapped[-1]
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        unwrapped.append(unwrapped[-1] + delta)
    return math.degrees(unwrapped[-1] - unwrapped[0])


def measure(frames):
    sequence = facts.actor_sequences(frames)["track"]
    return direction_action(sequence)


class ActionTests(unittest.TestCase):
    def test_ego_turning_does_not_become_actor_turning(self):
        for ego_turn in (math.radians(-30.0), math.radians(30.0)):
            with self.subTest(ego_turn=round(math.degrees(ego_turn), 1)):
                frames = scene(0.0, ego_turn, 0.0, 0.0)
                self.assertEqual(measure(frames), "straight")

    def test_both_turning_the_same_way_is_not_missed(self):
        for turn, expected in ((math.radians(-30.0), "turn_right"),
                               (math.radians(30.0), "turn_left")):
            with self.subTest(expected=expected):
                frames = scene(0.0, turn, 0.0, turn)
                self.assertEqual(measure(frames), expected)

    def test_actor_turning_with_a_straight_ego(self):
        for turn, expected in ((math.radians(30.0), "turn_left"),
                               (math.radians(-30.0), "turn_right")):
            with self.subTest(expected=expected):
                frames = scene(0.0, 0.0, 0.0, turn)
                self.assertEqual(measure(frames), expected)

    def test_actor_local_yaw_jumping_179_to_minus_179(self):
        # actor 179° -> 181°: the stored local yaw really jumps -358°, true change +2°
        frames = frames_with_headings(lambda i: 0.0,
                                      lambda i: math.radians(179.0 if i < 6 else 181.0))
        tracks = facts.actor_sequences(frames)
        sequence = tracks["track"]
        jump = math.degrees(sequence[6]["yaw"] - sequence[5]["yaw"])
        self.assertAlmostEqual(jump, -358.0, places=6)
        self.assertAlmostEqual(net_world_turn(sequence), 2.0, places=6)
        self.assertEqual(direction_action(sequence), "straight")

    def test_actor_local_yaw_jumping_minus_179_to_179(self):
        # actor -179° -> -181°: the stored local yaw really jumps +358°, true change -2°
        frames = frames_with_headings(lambda i: 0.0,
                                      lambda i: math.radians(-179.0 if i < 6 else -181.0))
        tracks = facts.actor_sequences(frames)
        sequence = tracks["track"]
        jump = math.degrees(sequence[6]["yaw"] - sequence[5]["yaw"])
        self.assertAlmostEqual(jump, 358.0, places=6)
        self.assertAlmostEqual(net_world_turn(sequence), -2.0, places=6)
        self.assertEqual(direction_action(sequence), "straight")

    def test_ego_crossing_pi_while_the_actor_keeps_its_world_heading(self):
        # actor world heading 0° throughout; the ego goes 179° -> 181°, so the local yaw
        # jumps +358° while the actor itself does not turn at all
        frames = frames_with_headings(lambda i: math.radians(179.0 if i < 6 else 181.0),
                                      lambda i: 0.0)
        tracks = facts.actor_sequences(frames)
        sequence = tracks["track"]
        jump = math.degrees(sequence[6]["yaw"] - sequence[5]["yaw"])
        self.assertAlmostEqual(jump, 358.0, places=6)
        self.assertAlmostEqual(net_world_turn(sequence), 0.0, places=6)
        self.assertEqual(direction_action(sequence), "straight")

    def test_gapped_observations_use_their_own_frame_pose(self):
        present = {3, 7, 11}
        ego_turn = math.radians(40.0)
        frames = scene(0.0, ego_turn, 0.0, math.radians(60.0), present=present)
        tracks = facts.actor_sequences(frames)
        sequence = tracks["track"]
        self.assertEqual([item["index"] for item in sequence], [3, 7, 11])
        for item in sequence:
            fraction = item["index"] / (N - 1)
            expected = math.radians(60.0) * fraction
            self.assertAlmostEqual(item["world_yaw"], expected, places=9)
            self.assertAlmostEqual(item["yaw"], expected - ego_turn * fraction, places=9)
        self.assertEqual(measure(frames), "turn_left")

    def test_pose_rejections(self):
        def corrupt(mutate):
            frames = scene(0.0, 0.0, 0.0, 0.0)
            mutate(frames[4])
            return frames

        cases = {
            "missing": lambda frame: frame.__setitem__("ego2global", None),
            "not numeric": lambda frame: frame.__setitem__("ego2global", "not a matrix"),
            "wrong shape": lambda frame: frame.__setitem__("ego2global", np.eye(3)),
            "nan": lambda frame: frame["ego2global"].__setitem__((0, 0), float("nan")),
            "infinity": lambda frame: frame["ego2global"].__setitem__((0, 3), float("inf")),
            "all zero": lambda frame: frame.__setitem__("ego2global", np.zeros((4, 4))),
            "degenerate scale": lambda frame: frame.__setitem__("ego2global", np.diag([0.0, 0.0, 0.0, 1.0])),
            "glide reflection": lambda frame: frame.__setitem__("ego2global", np.diag([1.0, 1.0, -1.0, 1.0])),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                with self.assertRaises(facts.FactsError):
                    facts.actor_sequences(corrupt(mutate))

    def test_valid_float32_pose_is_accepted(self):
        frames = scene(0.0, 0.0, 0.0, 0.0)
        for frame in frames:
            frame["ego2global"] = frame["ego2global"].astype(np.float32)
        sequence = facts.actor_sequences(frames)["track"]
        self.assertEqual(len(sequence), N)


class EntryPointTests(unittest.TestCase):
    """Drive the real facts() entry with file, lane, and map dependencies replaced."""

    @staticmethod
    def measured_lanes(*args, **kwargs):
        sequences, selected, violation = args[3], args[4], args[5]
        return {
            "available": True,
            "reason": None,
            "map_context": "t_junction",
            "ego_lane_keeping": "keep",
            "other_origin": "ahead_same_lane",
            "other_action": direction_action(sequences[selected], through=violation),
            "other_lane_keeping": "keep",
            "interaction": "rear_end",
            "evidence": {},
            "bev_image": "/tmp/tool01-bev-fixture.png",
        }

    def run_facts(self, frames):
        with mock.patch.object(facts, "scene_pickle", return_value=(frames, "synthetic-token")):
            with mock.patch.object(facts, "outcome", return_value=("collision", None, 1)):
                with mock.patch.object(facts, "analyze_lanes", side_effect=self.measured_lanes):
                    with mock.patch.object(facts, "frame_images", return_value=[]):
                        return facts.facts("archive", "token")

    def test_facts_entry_reports_the_actor_turn(self):
        frames = scene(0.0, math.radians(-30.0), 0.0, math.radians(-360.0), start=(4.0, 0.0, 0.0))
        result = self.run_facts(frames)
        self.assertEqual(result["attack"]["other_action"], "turn_right")

    def test_facts_entry_keeps_a_straight_actor_with_a_turning_ego(self):
        frames = scene(0.0, math.radians(30.0), 0.0, 0.0, start=(4.0, 0.0, 0.0))
        result = self.run_facts(frames)
        self.assertEqual(result["attack"]["other_action"], "straight")

if __name__ == "__main__":
    unittest.main()
