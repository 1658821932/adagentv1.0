"""Tool 01 must read the official score table through the shared resolver."""
import csv
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from common import pdm_scores  # noqa: E402
from tools.tool01_failure_analyzer import structure_facts as facts  # noqa: E402

HEADER = ["token", "step", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
          "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score",
          "first_violation_step"]
TOKEN = "2021.09.16.19.49.00_veh-42_00990_01609-4abb45d9ee1a5c5c"
OTHER = "2020.01.01.00.00.00_veh-1_00000_00001-aaaaaaaaaaaaaaaa"
SHORT = TOKEN[-16:]


def row(token, collision, departure, score, step):
    return [token, "11", str(collision), str(departure), "0", "0", "1", "1", str(score), str(step)]


def scene_pickle(archive, token=TOKEN):
    """Minimal scene metadata so the CLI reaches the score table."""
    directory = Path(archive) / "WE_output/openscene_format/meta_datas"
    directory.mkdir(parents=True, exist_ok=True)
    frames = [{"timestamp": 1_600_000_000_000_000 + index * 500_000,
               "ego2global": np.eye(4), "map_location": "map", "cams": {},
               "anns": {"track_tokens": [], "gt_names": [],
                        "gt_boxes": np.zeros((0, 7)), "gt_velocity_3d": np.zeros((0, 3))}}
              for index in range(2)]
    with (directory / f"{token}.pkl").open("wb") as stream:
        pickle.dump(frames, stream)


def table(archive, suffix, rows, omit=()):
    directory = Path(archive) / "WE_output/openscene_format"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pdm_scores.PREFIX}{suffix}.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([name for name in HEADER if name not in omit])
        writer.writerows(rows)
    return path


class TableSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.archive = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_reactive_table_wins_when_both_exist(self):
        table(self.archive, "_NR", [row(TOKEN, 1, 0, 0, 4)])
        table(self.archive, "_R", [row(TOKEN, 0, 1, 0, 8)])
        kind, matched, step = facts.outcome(self.archive, SHORT)
        self.assertEqual((kind, step), ("collision", 8))       # the _NR row says departure/4
        self.assertEqual(matched, TOKEN)

    def test_non_reactive_table_is_used_when_alone(self):
        table(self.archive, "_NR", [row(TOKEN, 1, 0, 0, 6)])   # departure only
        kind, matched, step = facts.outcome(self.archive, SHORT)
        self.assertEqual((kind, step), ("road_departure", 6))
        self.assertEqual(pdm_scores.score_file(self.archive).name, "all_scenes_pdm_averages_NR.csv")

    def test_other_suffix_follows_the_shared_resolver(self):
        table(self.archive, "_custom", [row(TOKEN, 0, 0, 0, 5)])
        kind, matched, step = facts.outcome(self.archive, SHORT)
        self.assertEqual((kind, step), ("collision_and_departure", 5))
        self.assertEqual(matched, TOKEN)
        self.assertEqual(pdm_scores.score_file(self.archive).name, "all_scenes_pdm_averages_custom.csv")

    def test_missing_table_raises_with_the_source(self):
        with self.assertRaises(facts.FactsError) as caught:
            facts.outcome(self.archive, SHORT)
        message = str(caught.exception)
        self.assertIn(self.archive, message)
        self.assertIn(pdm_scores.PREFIX, message)

    def test_reactive_table_without_the_row_does_not_fall_back(self):
        table(self.archive, "_NR", [row(TOKEN, 1, 0, 0, 4)])
        table(self.archive, "_R", [row(OTHER, 0, 1, 0, 8)])
        with self.assertRaises(facts.FactsError) as caught:
            facts.outcome(self.archive, SHORT)
        message = str(caught.exception)
        self.assertIn("_R.csv", message)
        self.assertIn("found 0", message)

    def test_field_parsing_and_classification_are_unchanged(self):
        cases = {"collision": (0, 1, 0), "road_departure": (1, 0, 0),
                 "collision_and_departure": (0, 0, 0)}
        for expected, (collision, departure, score) in cases.items():
            with self.subTest(expected=expected):
                table(self.archive, "_R", [row(TOKEN, collision, departure, score, 3)])
                self.assertEqual(facts.outcome(self.archive, SHORT)[0], expected)
        table(self.archive, "_R", [row(TOKEN, 1, 1, 1, 3)])      # zero score with neither
        with self.assertRaises(facts.FactsError):
            facts.outcome(self.archive, SHORT)

    def test_broken_field_names_the_table(self):
        table(self.archive, "_R", [row(TOKEN, 1, 1, 0, 3)], omit=("first_violation_step",))
        with self.assertRaises(facts.FactsError) as caught:
            facts.outcome(self.archive, SHORT)
        self.assertIn("_R.csv", str(caught.exception))
        self.assertIn("first_violation_step", str(caught.exception))


class CommandLineTests(unittest.TestCase):
    """The adapter runs the extractor as a script, so the shared module must import there."""

    def test_help_parses_with_the_shared_module_importable(self):
        script = ROOT / "tools/tool01_failure_analyzer/structure_facts.py"
        completed = subprocess.run([sys.executable, str(script), "--help"],
                                   capture_output=True, text=True, cwd="/", timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--archive", completed.stdout)
        self.assertNotIn("ModuleNotFoundError", completed.stderr)

    def run_cli(self, archive, token):
        script = ROOT / "tools/tool01_failure_analyzer/structure_facts.py"
        return subprocess.run([sys.executable, str(script), "--archive", str(archive), "--token", token],
                              capture_output=True, text=True, cwd="/", timeout=120)

    def test_cli_imports_the_shared_module(self):
        with tempfile.TemporaryDirectory() as archive:
            scene_pickle(archive)
            completed = self.run_cli(archive, SHORT)
        self.assertEqual(completed.returncode, 2, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "error")
        self.assertIn(pdm_scores.PREFIX, payload["reason"])
        self.assertNotIn("ModuleNotFoundError", completed.stderr)

    def test_cli_reports_the_resolved_table_on_a_bad_token(self):
        with tempfile.TemporaryDirectory() as archive:
            scene_pickle(archive)
            table(archive, "_R", [row(OTHER, 1, 0, 0, 4)])
            completed = self.run_cli(archive, SHORT)
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("_R.csv", json.loads(completed.stdout)["reason"])


if __name__ == "__main__":
    unittest.main()