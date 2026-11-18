"""Tool 01 must not read a missing speed as a stopped vehicle."""
import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.tool01_failure_analyzer import structure_facts as facts  # noqa: E402

N = 12


def ego_matrix():
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def frames_with_velocity(velocity, box=(6.0, 0.0, 0.0)):
    """One vehicle at a fixed ego-frame pose; the velocity rows are given per frame."""
    frames = []
    for index in range(N):
        rows = velocity[index] if isinstance(velocity, dict) else velocity
        frames.append({
            "timestamp": 1_600_000_000_000_000 + index * 500_000,
            "ego2global": ego_matrix(),
            "map_location": "synthetic",
            "cams": {},
            "anns": {
                "track_tokens": ["veh"],
                "gt_names": ["vehicle"],
                "gt_boxes": np.array([[box[0], box[1], 0.0, 4.6, 1.9, 1.6, box[2]]], dtype=float),
                "gt_velocity_3d": np.array(rows, dtype=float),
            },
        })
    return frames


MOVING = [[8.0, 0.0, 0.0]]
STOPPED = [[0.0, 0.0, 0.0]]


class SpeedValidationTests(unittest.TestCase):
    def test_all_zero_speeds_are_still_stop(self):
        sequence = facts.actor_sequences(frames_with_velocity(STOPPED))["veh"]
        self.assertEqual(facts.classify_action(sequence, "ahead_same_lane"), "stop")

    def test_valid_non_zero_speed_keeps_the_original_classification(self):
        frames = frames_with_velocity(MOVING)
        for index, frame in enumerate(frames):   # heading turns left while driving
            frame["anns"]["gt_boxes"][0][6] = math.radians(3.0 * index)
        sequence = facts.actor_sequences(frames)["veh"]
        self.assertEqual(facts.classify_action(sequence, "ahead_same_lane"), "turn_left")

    def assert_rejected(self, frames, *expected):
        with self.assertRaises(facts.FactsError) as caught:
            facts.actor_sequences(frames)
        for text in expected:
            self.assertIn(text, str(caught.exception))

    def test_missing_velocity_field_raises(self):
        frames = frames_with_velocity(MOVING)
        frames[4]["anns"].pop("gt_velocity_3d")
        self.assert_rejected(frames, "frame 4", "veh")

    def test_missing_velocity_row_raises(self):
        self.assert_rejected(frames_with_velocity([]), "frame 0", "veh")

    def test_short_velocity_rows_raise(self):
        for rows in ([[8.0]], np.zeros((1, 1)), np.zeros((1, 3, 1))):
            with self.subTest(shape=np.shape(rows)):
                self.assert_rejected(frames_with_velocity(rows), "frame 0", "veh")

    def test_any_frame_without_velocity_raises(self):
        for column in (0, 1):
            with self.subTest(column=column):
                frames = frames_with_velocity(MOVING)
                frames[5]["anns"]["gt_velocity_3d"][0][column] = float("nan")
                self.assert_rejected(frames, "frame 5", "veh")

    def test_infinite_velocity_component_raises(self):
        for column in (0, 1):
            with self.subTest(column=column):
                frames = frames_with_velocity(MOVING)
                frames[6]["anns"]["gt_velocity_3d"][0][column] = float("inf")
                self.assert_rejected(frames, "frame 6", "veh")


class ClassificationSpeedTests(unittest.TestCase):
    def sequence(self, **overrides):
        item = dict(index=0, x=6.0, y=0.0, length=4.6, width=1.9, yaw=0.0, world_yaw=0.0, speed=8.0)
        item.update(overrides)
        for name in [name for name, value in overrides.items() if value is _MISSING]:
            item.pop(name)
        return [item, dict(item, index=1)]

    def assert_rejected(self, sequence):
        with self.assertRaises(facts.FactsError):
            facts.classify_action(sequence, "ahead_same_lane")

    def test_missing_speed_is_not_stop(self):
        self.assert_rejected(self.sequence(speed=_MISSING))

    def test_none_speed_is_not_stop(self):
        self.assert_rejected(self.sequence(speed=None))

    def test_non_finite_speed_is_not_stop(self):
        for speed in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(speed=speed):
                self.assert_rejected(self.sequence(speed=speed))

    def test_missing_speed_in_a_later_frame_is_not_stop(self):
        sequence = self.sequence()
        sequence[1].pop("speed")
        self.assert_rejected(sequence)


class _Missing:
    pass


_MISSING = _Missing()


class SpeedEntryPointTests(unittest.TestCase):
    def test_broken_velocity_does_not_become_a_normal_result(self):
        frames = frames_with_velocity(MOVING)
        frames[3]["anns"]["gt_velocity_3d"][0][1] = float("nan")
        with mock.patch.object(facts, "scene_pickle", return_value=(frames, "synthetic")), \
             mock.patch.object(facts, "outcome", return_value=("collision", None, 7)), \
             mock.patch.object(facts, "map_context", return_value="t_junction"), \
             mock.patch.object(facts, "frame_images", return_value=[]):
            with self.assertRaises(facts.FactsError):
                facts.facts("archive", "token")

    def test_complete_velocity_still_produces_a_result(self):
        frames = frames_with_velocity(MOVING)
        with mock.patch.object(facts, "scene_pickle", return_value=(frames, "synthetic")), \
             mock.patch.object(facts, "outcome", return_value=("collision", None, 7)), \
             mock.patch.object(facts, "map_context", return_value="t_junction"), \
             mock.patch.object(facts, "frame_images", return_value=[]):
            result = facts.facts("archive", "token")
        self.assertEqual(result["attack"]["other_origin"], "ahead_same_lane")
        self.assertEqual(result["attack"]["other_action"], "straight")


if __name__ == "__main__":
    unittest.main()