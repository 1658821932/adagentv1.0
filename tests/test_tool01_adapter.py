"""The adapter must reject broken extractor runs instead of faking success."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp_servers.worldengine_failure_analysis.adapter import service  # noqa: E402

FACTS = {"status": "ok", "token": "scene-4abb45d9ee1a5c5c", "map_location": "map",
         "violation_step": 8, "failure": {"outcome": "collision"},
         "attack": {"map_context": "t_junction", "ego_action": "turn_right", "ego_lane_keeping": "keep",
                    "other_origin": "right_arm", "other_action": "turn_left", "other_lane_keeping": "keep"},
         "environment": {"time": "2021-09-16T20:16:41+00:00", "weather": None, "extra": None},
         "frame_images": [{"index": 0, "time_s": 0.0, "images": {}}],
         "bev_image": "/tmp/tool01-bev-fixture.png"}


class AdapterFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.archive = Path(self.tmp.name) / "archive"
        self.archive.mkdir()
        (self.archive / "scene_index.csv").write_text("token\nfull-token\n")
        self.patches = [mock.patch.object(service, "resolve_simulation", return_value=self.archive),
                        mock.patch.object(service, "_resolve_python", return_value=sys.executable)]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def call(self, run_impl):
        with mock.patch.object(service.subprocess, "run", run_impl):
            return service.failure_facts("full-token", "fixture/archive", timeout_s=30)

    def assert_rejected(self, run_impl, *expected):
        with self.assertRaises(service.WfaError) as caught:
            self.call(run_impl)
        for text in expected:
            self.assertIn(text, str(caught.exception))
        return caught.exception

    def test_timeout_keeps_the_stderr_tail(self):
        def slow(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout"), stderr="partial stderr")
        error = self.assert_rejected(slow, "timed out")
        self.assertTrue(error.details.get("timeout"))
        self.assertEqual(error.details.get("stage"), "data")

    def test_os_error_is_reported(self):
        def broken(command, **kwargs):
            raise OSError("cannot execute")
        error = self.assert_rejected(broken, "could not start")
        self.assertFalse(error.details.get("timeout"))

    def test_non_zero_exit_is_not_success_even_with_ok_stdout(self):
        def failing(command, **kwargs):
            return subprocess.CompletedProcess(command, 2, json.dumps(FACTS), "boom: missing lib")
        self.assert_rejected(failing, "exited with 2", "boom")

    def test_non_zero_exit_keeps_the_reported_reason(self):
        payload = {"status": "error",
                   "reason": "intersection 17562 in map us-pa-pittsburgh-hazelwood connects 1 road segments"}
        def failing(command, **kwargs):
            return subprocess.CompletedProcess(command, 2, json.dumps(payload), "")
        error = self.assert_rejected(failing, "exited with 2", "17562")
        self.assertEqual(error.details.get("returncode"), 2)

    def test_non_zero_exit_appends_the_stderr_tail(self):
        payload = {"status": "error", "reason": "frame 5 velocity for veh has non-finite components"}
        def failing(command, **kwargs):
            return subprocess.CompletedProcess(command, 2, json.dumps(payload), "traceback tail")
        self.assert_rejected(failing, "frame 5", "traceback tail")

    def test_empty_stdout_is_reported(self):
        def silent(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "   ", "warned")
        self.assert_rejected(silent, "no output", "warned")

    def test_bad_json_is_reported(self):
        def garbage(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "not json", "")
        self.assert_rejected(garbage, "invalid JSON")

    def test_json_array_is_reported(self):
        def array(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "[1, 2, 3]", "")
        self.assert_rejected(array, "failure fact extraction failed")

    def test_error_status_is_reported(self):
        def reported(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, json.dumps({"status": "error",
                                                                      "reason": "no scene pickle"}), "")
        self.assert_rejected(reported, "no scene pickle")

    def test_missing_required_fields_do_not_raise_key_error(self):
        for key in ("token", "map_location", "violation_step", "failure", "attack",
                    "environment", "frame_images"):
            with self.subTest(key=key):
                payload = {name: value for name, value in FACTS.items() if name != key}
                def incomplete(command, _payload=payload, **kwargs):
                    return subprocess.CompletedProcess(command, 0, json.dumps(_payload), "")
                self.assert_rejected(incomplete, "incomplete facts", key)

    def test_non_text_token_is_reported(self):
        payload = dict(FACTS, token=1234)
        def numeric(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        self.assert_rejected(numeric, "non-text token")

    def test_complete_payload_is_enriched(self):
        def complete(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, json.dumps(FACTS), "")
        payload = self.call(complete)
        self.assertEqual(payload["archive"], str(self.archive.resolve()))
        self.assertTrue(payload["images_dir"].endswith("sensor_blobs/4abb45d9ee1a5c5c"))


if __name__ == "__main__":
    unittest.main()
