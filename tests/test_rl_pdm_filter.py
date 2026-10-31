"""Scene-level PDM-zero failure filtering for RL training preparation."""
import tempfile
import unittest
from pathlib import Path

from mcp_servers.worldengine.adapter.pdm_filter import load_scene_scores, failed_scene_anchors

HEADER = ("token,step,no_at_fault_collisions,drivable_area_compliance,ego_progress,"
          "time_to_collision_within_bound,comfort,driving_direction_compliance,"
          "score,first_violation_step\n")


def write_scores(root, rows, react_type="NR"):
    path = Path(root) / f"all_scenes_pdm_averages_{react_type}.csv"
    path.write_text(HEADER + rows)
    return path


class PdmFilterTests(unittest.TestCase):
    def test_zero_score_scene_is_selected(self):
        with tempfile.TemporaryDirectory() as d:
            write_scores(d,
                "log_a-aaa,11,1,1,0,1,1,1,0.0,\n"
                "log_b-bbb,11,1,1,0.4,1,1,1,0.76,\n"
                "overall_average,11,1,1,0.2,1,1,1,0.38,\n")
            scores, _ = load_scene_scores(d)
            self.assertEqual(scores, {"log_a-aaa": 0.0, "log_b-bbb": 0.76})
            self.assertEqual(failed_scene_anchors(scores, {"aaa", "bbb"}), {"aaa"})

    def test_scenes_outside_requested_tokens_are_ignored(self):
        scores = {"log_a-aaa": 0.0, "log_c-ccc": 0.0}
        self.assertEqual(failed_scene_anchors(scores, {"aaa"}), {"aaa"})

    def test_multi_part_log_names_use_the_final_anchor(self):
        scores = {"2021.06.09-x-3f4d95a745ac552a": 0.0}
        self.assertEqual(failed_scene_anchors(scores, {"3f4d95a745ac552a"}),
                         {"3f4d95a745ac552a"})

    def test_no_zero_score_returns_empty(self):
        scores = {"log_a-aaa": 0.5, "log_b-bbb": 0.9}
        self.assertEqual(failed_scene_anchors(scores, {"aaa", "bbb"}), set())

    def test_missing_scores_file(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_scene_scores(d)

    def test_multiple_react_type_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            write_scores(d, "log_a-aaa,11,1,1,0,1,1,1,0.0,\n", "NR")
            write_scores(d, "log_a-aaa,11,1,1,0,1,1,1,0.0,\n", "R")
            with self.assertRaisesRegex(ValueError, "Multiple PDM average files"):
                load_scene_scores(d)

    def test_invalid_score_value(self):
        with tempfile.TemporaryDirectory() as d:
            write_scores(d, "log_a-aaa,11,1,1,0,1,1,1,not_a_number,\n")
            with self.assertRaisesRegex(ValueError, "Invalid PDM score"):
                load_scene_scores(d)


if __name__ == "__main__":
    unittest.main()
