"""The public run() must enforce its budget as a hard, group-wide stop."""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.tool01_failure_analyzer import evidence, real  # noqa: E402

TOKEN = "4abb45d9ee1a5c5c"
SIMULATION = "experiment/batch"

STUB_WORKER = '''
import json, sys
from unittest import mock
sys.path.insert(0, {root!r})
from tools.tool01_failure_analyzer import real

request = json.load(sys.stdin)
facts = dict(archive="archive", token="scene-4abb45d9ee1a5c5c", simulation_record="experiment/batch",
             map_location="map", violation_step=7,
             failure={{"outcome": "collision", "interaction": "crossing", "ego_cause": "x"}},
             attack={{"map_context": "t_junction", "ego_action": "turn_right",
                      "other_origin": "right_arm", "other_action": "turn_left"}},
             environment={{"time": "2021-09-16T20:16:41+00:00", "weather": None, "extra": None}})
visual = dict(matched=True, notes=[], weather="clear", extra="")
with mock.patch.object(real, "failure_facts", return_value=facts), \\
     mock.patch.object(real, "visual_check", return_value=visual):
    result = real.analyze(request["input_data"], request["run_dir"])
json.dump(result, sys.stdout)
'''

BLOCKING_WORKER = '''
import subprocess, sys, time
subprocess.Popen([sys.executable, "-c",
                  "import sys, time; time.sleep(5); open(sys.argv[1], 'w').write('late')",
                  {late!r}])
time.sleep(30)
'''


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs = self.root / "runs"
        self.original = real.worker_command

    def tearDown(self):
        real.worker_command = self.original
        self.tmp.cleanup()

    def script(self, name, source):
        path = self.root / name
        path.write_text(source)
        return [sys.executable, str(path)]

    def call(self, timeout, command=None):
        if command is not None:
            real.worker_command = lambda: command
        return real.run({"token": TOKEN, "simulation": SIMULATION, "timeout": timeout},
                        runs_root=self.runs)

    def runs_dirs(self):
        return sorted(path for path in self.runs.rglob("manifest.json"))

    def test_success_runs_in_a_real_child_process(self):
        result = self.call(120, self.script("stub_worker.py", STUB_WORKER.format(root=str(ROOT))))
        self.assertEqual(result["status"], "success")
        evidence_dir = Path(result["result"]["evidence_dir"])
        self.assertTrue(evidence_dir.is_dir())
        manifest = evidence.read_manifest(evidence_dir)
        self.assertEqual((manifest["status"], manifest["stage"]), ("success", "done"))
        self.assertEqual(manifest["simulation"], SIMULATION)
        self.assertEqual(manifest["token"], TOKEN)
        self.assertEqual(manifest["resolved_token"], "scene-4abb45d9ee1a5c5c")
        self.assertTrue((evidence_dir / "worker.log").is_file())
        self.assertTrue((evidence_dir / "request.json").is_file())

    def test_timeout_kills_the_group_and_freezes_the_evidence(self):
        late = self.root / "late.txt"
        command = self.script("blocking_worker.py", BLOCKING_WORKER.format(late=str(late)))
        started = time.monotonic()
        result = self.call(2, command)
        elapsed = time.monotonic() - started
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "timeout")
        self.assertLess(elapsed, 20)          # one grace period at most, no full sleep
        evidence_dir = Path(result["error"]["evidence_dir"])
        manifest = evidence.read_manifest(evidence_dir)
        self.assertEqual((manifest["status"], manifest["error"]), ("failed", "timeout"))
        self.assertEqual(manifest["timeout_s"], 2.0)

        before = {path: path.stat().st_mtime_ns for path in evidence_dir.rglob("*") if path.is_file()}
        time.sleep(6.5)                        # the grandchild would write at 5s if it survived
        self.assertFalse(late.exists(), "the grandchild process outlived the timeout")
        after = {path: path.stat().st_mtime_ns for path in evidence_dir.rglob("*") if path.is_file()}
        self.assertEqual(before, after, "the evidence directory kept changing after the timeout")

    def test_non_zero_exit_is_not_success(self):
        command = self.script("failing_worker.py",
                              "import sys\nsys.stderr.write('boom')\nsys.exit(3)\n")
        result = self.call(60, command)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "analysis_error")
        self.assertIn("3", result["error"]["message"])
        manifest = evidence.read_manifest(Path(result["error"]["evidence_dir"]))
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("boom", json.dumps(manifest))

    def test_unparseable_output_is_not_success(self):
        command = self.script("garbage_worker.py", "print('not json')\n")
        result = self.call(60, command)
        self.assertEqual(result["error"]["type"], "analysis_error")
        self.assertIn("JSON", result["error"]["message"])

    def test_unexpected_result_shape_is_not_success(self):
        command = self.script("shape_worker.py", "print('{\"status\": \"weird\"}')\n")
        result = self.call(60, command)
        self.assertEqual(result["error"]["type"], "analysis_error")
        self.assertIn("shape", result["error"]["message"])

    def test_worker_that_cannot_start_is_structured(self):
        result = self.call(30, ["/nonexistent/tool01-worker"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "analysis_error")
        self.assertEqual(result["error"]["stage"], "start")
        manifest = evidence.read_manifest(Path(result["error"]["evidence_dir"]))
        self.assertEqual((manifest["status"], manifest["stage"]), ("failed", "start"))

    def test_success_without_a_valid_result_is_not_success(self):
        for payload in ('{"status": "success", "tool": "failure_analyzer"}',
                        '{"status": "success", "tool": "failure_analyzer", "result": {}}',
                        '{"status": "success", "tool": "failure_analyzer", "result": {"token": 5}}'):
            with self.subTest(payload=payload):
                command = self.script("bad_success.py", f"print({payload!r})\n")
                result = self.call(30, command)
                self.assertEqual(result["error"]["type"], "analysis_error")
                self.assertIn("contract", result["error"]["message"])

    def test_success_from_another_tool_identity_is_not_success(self):
        command = self.script("other_tool.py",
                              'import json\nprint(json.dumps({"status": "success", "tool": "other_tool"}))\n')
        result = self.call(30, command)
        self.assertEqual(result["error"]["type"], "analysis_error")
        self.assertIn("tool identity", result["error"]["message"])

    def test_malformed_error_payload_is_not_success(self):
        for payload in ('{"status": "error", "tool": "failure_analyzer", "error": "boom"}',
                        '{"status": "error", "tool": "failure_analyzer", "error": {"type": ""}}'):
            with self.subTest(payload=payload):
                command = self.script("bad_error.py", f"print({payload!r})\n")
                result = self.call(30, command)
                self.assertEqual(result["error"]["type"], "analysis_error")
                self.assertIn("error payload", result["error"]["message"])

    def test_default_worker_reports_a_missing_simulation_from_the_data_stage(self):
        missing = str(self.root / "no-such-archive")
        result = real.run({"token": "4abb45d9ee1a5c5c", "simulation": missing, "timeout": 60},
                          runs_root=self.runs)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "failure_facts_error")
        self.assertEqual(result["error"]["stage"], "data")
        evidence_dir = Path(result["error"]["evidence_dir"])
        self.assertTrue(evidence_dir.is_dir())
        manifest = evidence.read_manifest(evidence_dir)
        self.assertEqual((manifest["status"], manifest["stage"]), ("failed", "data"))
        self.assertTrue((evidence_dir / "worker.log").is_file())

    def test_missing_input_is_reported_without_spawning_a_run(self):
        result = real.run({"timeout": 30}, runs_root=self.runs)
        self.assertEqual(result["error"]["type"], "input_error")
        self.assertEqual(result["error"]["evidence_dir"], "")
        self.assertEqual(self.runs_dirs(), [])


if __name__ == "__main__":
    unittest.main()