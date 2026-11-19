"""Target selection: closest body within 5 m of the ego body at the violation frame.

The selected vehicle is the analysis target of this rule; it is not an officially
identified collision or cause object.
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

N = 12
REAR_AXLE_TO_FRONT = None      # filled from the scorer parameters in setUpModule
EGO_LENGTH = None


def ego_matrix(x=0.0, y=0.0, yaw=0.0):
    cos, sin = math.cos(yaw), math.sin(yaw)
    return np.array([[cos, -sin, 0.0, x],
                     [sin, cos, 0.0, y],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def setUpModule():
    global REAR_AXLE_TO_FRONT, EGO_LENGTH
    EGO_LENGTH, _, shift = facts.ego_geometry()
    REAR_AXLE_TO_FRONT = shift + EGO_LENGTH / 2.0


def frames_with(vehicles_at):
    """vehicles_at: index -> list of (track, x, y, length, width, yaw_deg)."""
    frames = []
    for index in range(N):
        rows = vehicles_at.get(index, [])
        frames.append({
            "timestamp": 1_600_000_000_000_000 + index * 500_000,
            "ego2global": ego_matrix(),
            "map_location": "synthetic",
            "cams": {},
            "anns": {
                "track_tokens": [row[0] for row in rows],
                "gt_names": ["vehicle"] * len(rows),
                "gt_boxes": np.array([[row[1], row[2], 0.0, row[3], row[4], 1.6,
                                       math.radians(row[5])] for row in rows],
                                     dtype=float).reshape(-1, 7),
                "gt_velocity_3d": np.array([[8.0, 0.0, 0.0]] * len(rows), dtype=float).reshape(-1, 3),
            },
        })
    return frames


def ahead(center_x, length=4.6, width=1.9, track="veh", y=0.0, yaw=0.0):
    return (track, center_x, y, length, width, yaw)


def gap_for(center_x, length):
    """Body gap when the vehicle stands straight ahead of the ego."""
    return center_x - length / 2.0 - REAR_AXLE_TO_FRONT


def select(frames, step):
    return facts.conflicting_track(facts.actor_sequences(frames), step, frames)


class TargetSelectionTests(unittest.TestCase):
    def test_ego_geometry_comes_from_the_scorer_parameters(self):
        import worldengine.components.agents.vehicle_model.pacifica_vehicle as pacifica

        parameters = pacifica.get_pacifica_parameters()
        self.assertEqual(facts.ego_geometry(),
                         (float(parameters.length), float(parameters.width),
                          float((parameters.front_length - parameters.rear_length) / 2.0)))

    def test_closest_body_within_the_limit_wins(self):
        near = ahead(8.0, track="near")     # gap 1.65 m
        far = ahead(11.0, track="far")      # gap 4.65 m
        self.assertAlmostEqual(gap_for(8.0, 4.6), 1.651, places=6)
        self.assertAlmostEqual(gap_for(11.0, 4.6), 4.651, places=6)
        name, _ = select(frames_with({7: [far, near]}), 7)
        self.assertEqual(name, "near")

    def test_the_limit_is_inclusive_and_exclusive_just_beyond(self):
        exact = 5.0 + REAR_AXLE_TO_FRONT + 4.6 / 2.0
        self.assertAlmostEqual(gap_for(exact, 4.6), 5.0, places=9)
        self.assertEqual(select(frames_with({7: [ahead(exact)]}), 7)[0], "veh")
        self.assertEqual(select(frames_with({7: [ahead(exact + 0.01)]}), 7), (None, None))

    def test_body_distance_beats_centre_distance(self):
        bus = ahead(12.0, length=12.0, width=2.55, track="bus")   # centre 12.0, gap 1.95
        car = ahead(9.0, length=4.4, width=1.8, track="car")      # centre  9.0, gap 2.75
        self.assertGreater(math.hypot(12.0, 0.0), math.hypot(9.0, 0.0))
        self.assertLess(gap_for(12.0, 12.0), gap_for(9.0, 4.4))
        self.assertEqual(select(frames_with({7: [car, bus]}), 7)[0], "bus")

    def test_no_vehicle_or_nothing_within_the_limit_gives_no_target(self):
        harness_limit = 5.0 + REAR_AXLE_TO_FRONT + 4.6 / 2.0
        self.assertEqual(select(frames_with({}), 7), (None, None))
        self.assertEqual(select(frames_with({7: [ahead(30.0)]}), 7), (None, None))
        self.assertEqual(select(frames_with({7: [ahead(harness_limit + 0.5)]}), 7), (None, None))

    def test_non_zero_violation_frame_uses_that_frame_only(self):
        vehicles = {index: [ahead(30.0)] for index in range(N)}
        vehicles[7] = [ahead(6.0)]
        frames = frames_with(vehicles)
        self.assertEqual(select(frames, 7)[0], "veh")
        self.assertEqual(select(frames, 8), (None, None))

    def test_ties_are_broken_by_track_token(self):
        first = ahead(8.0, track="alpha")
        second = ahead(8.0, track="beta")
        self.assertEqual(select(frames_with({7: [second, first]}), 7)[0], "alpha")

    def test_invalid_violation_frame_raises(self):
        frames = frames_with({7: [ahead(6.0)]})
        for step in (-1, N, 7.5, "7", None, True):
            with self.subTest(step=step):
                with self.assertRaises(facts.FactsError):
                    select(frames, step)

    def test_unusable_geometry_raises(self):
        for box in (ahead(6.0, length=0.0), ahead(6.0, width=-1.0), ahead(6.0, width=0.0)):
            with self.subTest(box=box):
                with self.assertRaises(facts.FactsError):
                    select(frames_with({7: [box]}), 7)


class VehicleGeometryValidationTests(unittest.TestCase):
    """Broken vehicle boxes must raise, never turn into "no target vehicle"."""

    def broken(self, mutate):
        frames = frames_with({7: [ahead(6.0)]})
        mutate(frames[7]["anns"])
        return frames

    def assert_rejected(self, frames, *expected):
        with self.assertRaises(facts.FactsError) as caught:
            facts.actor_sequences(frames)
        for text in expected:
            self.assertIn(text, str(caught.exception))

    def test_non_finite_position_raises(self):
        for column in (0, 1):
            with self.subTest(column=column):
                frames = self.broken(lambda anns: anns["gt_boxes"].__setitem__((0, column), float("nan")))
                self.assert_rejected(frames, "frame 7", "veh")

    def test_infinite_size_raises(self):
        for column in (3, 4):
            with self.subTest(column=column):
                frames = self.broken(lambda anns: anns["gt_boxes"].__setitem__((0, column), float("inf")))
                self.assert_rejected(frames, "frame 7", "veh")

    def test_non_finite_heading_raises(self):
        frames = self.broken(lambda anns: anns["gt_boxes"].__setitem__((0, 6), float("nan")))
        self.assert_rejected(frames, "frame 7", "veh")

    def test_missing_or_short_box_raises(self):
        cases = {
            "missing": lambda anns: anns.pop("gt_boxes"),
            "short": lambda anns: anns.__setitem__("gt_boxes", np.zeros((1, 5))),
            "fewer rows than tokens": lambda anns: anns.__setitem__("gt_boxes", np.zeros((0, 7))),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                self.assert_rejected(self.broken(mutate), "frame 7", "veh")

    def test_empty_scene_still_reports_no_target(self):
        frames = frames_with({})
        self.assertEqual(facts.actor_sequences(frames), {})
        self.assertEqual(select(frames, 7), (None, None))

    def test_broken_box_does_not_become_no_target_through_facts(self):
        frames = frames_with({7: [ahead(6.0)]})
        frames[7]["anns"]["gt_boxes"][0][0] = float("nan")
        with mock.patch.object(facts, "scene_pickle", return_value=(frames, "synthetic")), \
             mock.patch.object(facts, "outcome", return_value=("collision", None, 7)), \
             mock.patch.object(facts, "map_context", return_value="t_junction"), \
             mock.patch.object(facts, "frame_images", return_value=[]):
            with self.assertRaises(facts.FactsError):
                facts.facts("archive", "token")


class TargetEntryPointTests(unittest.TestCase):
    def call(self, frames, lane_data=None):
        original = facts.analyze_lanes

        def measured(*args, **kwargs):
            return dict(lane_data) if lane_data is not None else original(*args, **kwargs)

        with mock.patch.object(facts, "scene_pickle", return_value=(frames, "synthetic")):
            with mock.patch.object(facts, "outcome", return_value=("collision", None, 7)):
                with mock.patch.object(facts, "analyze_lanes", side_effect=measured):
                    with mock.patch.object(facts, "map_ego_action", return_value="straight"):
                        with mock.patch.object(facts, "frame_images", return_value=[]):
                            return facts.facts("archive", "token")

    def test_facts_reports_no_target_and_says_so(self):
        lane_data = {
            "available": True,
            "reason": None,
            "map_context": "t_junction",
            "ego_lane_keeping": "keep",
            "other_origin": None,
            "other_action": None,
            "other_lane_keeping": "unknown",
            "interaction": "single_vehicle",
            "evidence": {},
        }
        result = self.call(frames_with({7: [ahead(30.0)]}), lane_data=lane_data)
        self.assertEqual(result["attack"]["other_origin"], "none")
        self.assertEqual(result["attack"]["other_action"], "none")
        self.assertEqual(result["failure"]["interaction"], "single_vehicle")
        self.assertEqual(result["failure"]["ego_cause"],
                         "在丁字路口直行并保持在本车道内时发生单车碰撞，首次违规时刻5米内无目标车辆")

    def test_selected_target_without_an_official_lane_binding_is_an_error(self):
        with self.assertRaises(facts.FactsError) as caught:
            self.call(frames_with({7: [ahead(6.0)]}))
        self.assertIn("lane analysis", str(caught.exception))

if __name__ == "__main__":
    unittest.main()
