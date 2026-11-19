"""Every call keeps its own evidence directory with the frames it really sent."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.tool01_failure_analyzer import evidence, real, visual_check  # noqa: E402

COUNT = 12
VIOLATION = 7


def camera_frames(count=COUNT, root=None):
    root = Path(root) / "sensor_blobs" / "scene"
    images = {}
    for camera in visual_check.CAMERAS:
        (root / camera).mkdir(parents=True, exist_ok=True)
        images[camera] = []
        for index in range(count):
            path = root / camera / f"{index:02d}.jpg"
            Image.new("RGB", (64, 36), (index * 7 % 256, 80, 140)).save(path, format="JPEG", quality=90)
            images[camera].append(path)
    return [{"index": index, "time_s": index * 0.5,
             "images": {camera: str(images[camera][index]) for camera in visual_check.CAMERAS}}
            for index in range(count)]


def facts_for(archive="archive", simulation="run/sim", images=None, root=None):
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    bev = base / "bev.png"
    Image.new("RGB", (32, 24), (20, 80, 120)).save(bev, format="PNG")
    return {"archive": archive, "token": "2021.09.16.19.49.00_veh-42_00990_01609-4abb45d9ee1a5c5c",
            "simulation_record": simulation, "map_location": "map", "violation_step": VIOLATION,
            "failure": {"outcome": "collision", "interaction": "crossing", "ego_cause": "x"},
            "attack": {"map_context": "t_junction", "ego_action": "turn_right", "ego_lane_keeping": "keep",
                       "other_origin": "right_arm", "other_action": "turn_left", "other_lane_keeping": "keep"},
            "environment": {"time": "2021-09-16T20:16:41+00:00", "weather": None, "extra": None},
            "frame_images": camera_frames(root=root) if images is None else images,
            "bev_image": str(bev)}


FINAL = {"weather": "cloudy", "extra": "白天", "notes": []}
BEV_FINAL = {"map_context": "t_junction", "notes": []}


class StubClient:
    answer = json.dumps(FINAL, ensure_ascii=False)
    bev_answer = json.dumps(BEV_FINAL, ensure_ascii=False)
    calls = []
    inspect = None

    def __init__(self, component=None):
        self.component = component

    def chat(self, messages):
        StubClient.calls.append(messages)
        if StubClient.inspect is not None:
            StubClient.inspect()
        content = messages[1]["content"]
        if content and content[0].get("type") == "text" and "BEV" in content[0].get("text", ""):
            return self.bev_answer
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def analyze(run_dir, facts=None, client=StubClient):
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with mock.patch.object(real, "failure_facts",
                           return_value=facts or facts_for(root=directory.parent)), \
         mock.patch("agent.llm_client.LLMClient", client):
        return real.analyze({"token": "4abb45d9ee1a5c5c", "simulation": "run/sim", "timeout": 120},
                            directory)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run"

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_records_frames_sources_and_real_hashes(self):
        result = analyze(self.run_dir)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["evidence_dir"], str(self.run_dir))

        manifest = evidence.read_manifest(self.run_dir)
        self.assertEqual((manifest["status"], manifest["stage"]), ("success", "done"))
        indices = [VIOLATION]
        self.assertEqual([frame["index"] for frame in manifest["frames"]], indices)
        self.assertEqual([entry["index"] for entry in manifest["selected"]], indices)
        for entry in manifest["selected"]:
            self.assertEqual(len(entry["sources"]), 8)
            self.assertTrue(Path(entry["sources"]["CAM_F0"]).is_file())
        for frame in manifest["frames"]:
            sheet = Path(frame["sheet"])
            self.assertTrue(sheet.is_file())
            self.assertEqual(frame["sheet_sha256"], evidence.sha256_file(sheet))

        sent = manifest["sent_images"]
        self.assertEqual([entry["index"] for entry in sent], indices)
        content = StubClient.calls[-1][1]["content"]
        urls = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
        self.assertEqual([entry["sha256"] for entry in sent],
                         [evidence.sent_image_sha256(url) for url in urls])
        # the sent bytes are a re-encoded crop, so they must not equal the camera file hash
        original = evidence.sha256_file(manifest["selected"][0]["sources"]["CAM_F0"])
        self.assertNotEqual(sent[0]["sha256"], original)
        self.assertTrue((self.run_dir / "response.txt").is_file())

    def test_model_timeout_keeps_measured_data_and_degrades_environment(self):
        StubClient.answer = TimeoutError("request timed out")
        try:
            result = analyze(self.run_dir)
        finally:
            StubClient.answer = json.dumps(FINAL, ensure_ascii=False)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["attack"]["ego_action"], "turn_right")
        self.assertEqual(result["result"]["environment"]["weather"], "unknown")
        self.assertEqual(result["result"]["environment"]["extra"], "")
        self.assertTrue(any("环境补充失败" in note for note in result["result"]["verification"]["notes"]))
        manifest = evidence.read_manifest(self.run_dir)
        self.assertEqual((manifest["status"], manifest["stage"]), ("success", "done"))
        self.assertIn("optional_error", manifest)
        self.assertEqual(len(manifest["frames"]), 1)

    def test_empty_reply_degrades_only_environment(self):
        StubClient.answer = RuntimeError("LLM returned empty response")
        try:
            result = analyze(self.run_dir)
        finally:
            StubClient.answer = json.dumps(FINAL, ensure_ascii=False)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["environment"]["weather"], "unknown")
        self.assertEqual(result["result"]["environment"]["extra"], "")

    def test_visual_model_cannot_change_key_data(self):
        invalid = dict(FINAL, ego_action="straight", interaction="opposite_direction")
        StubClient.answer = json.dumps(invalid, ensure_ascii=False)
        try:
            result = analyze(self.run_dir)
        finally:
            StubClient.answer = json.dumps(FINAL, ensure_ascii=False)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["attack"]["ego_action"], "turn_right")
        self.assertEqual(result["result"]["failure"]["interaction"], "crossing")
        self.assertEqual(result["result"]["verification"]["corrections"], [])
        self.assertEqual(result["result"]["environment"]["weather"], "unknown")
        self.assertEqual(evidence.read_manifest(self.run_dir)["status"], "success")

    def test_bev_road_context_correction_rebuilds_the_cause(self):
        StubClient.bev_answer = json.dumps(
            {"map_context": "curve", "notes": ["车道连续弯曲"]}, ensure_ascii=False)
        try:
            result = analyze(self.run_dir)
        finally:
            StubClient.bev_answer = json.dumps(BEV_FINAL, ensure_ascii=False)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["attack"]["map_context"], "curve")
        self.assertEqual(result["result"]["verification"]["corrections"], [
            {"field": "map_context", "measured": "t_junction", "final": "curve"},
        ])
        self.assertTrue(result["result"]["failure"]["ego_cause"].startswith("在弯道上"))

    def test_data_failure_is_reported_with_its_stage(self):
        with mock.patch.object(real, "failure_facts",
                               side_effect=real.WfaError("no archive", stage="data", timeout=True)):
            result = real.analyze({"token": "t", "simulation": "s", "timeout": 60}, self.run_dir)
        self.assertEqual((result["error"]["type"], result["error"]["stage"]), ("timeout", "data"))
        self.assertEqual(evidence.read_manifest(self.run_dir)["stage"], "data")


class AnalysisStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run"
        StubClient.inspect = None
        StubClient.answer = json.dumps(FINAL, ensure_ascii=False)

    def tearDown(self):
        StubClient.inspect = None
        self.tmp.cleanup()

    def test_stage_is_vlm_before_the_request_runs(self):
        seen = {}
        StubClient.inspect = lambda: seen.update(evidence.read_manifest(self.run_dir))
        analyze(self.run_dir)
        self.assertEqual(seen["stage"], "vlm")

    def test_selected_frames_are_saved_before_composition(self):
        with mock.patch.object(visual_check, "contact_sheet", side_effect=OSError("disk full")):
            result = analyze(self.run_dir)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["environment"]["weather"], "unknown")
        manifest = evidence.read_manifest(self.run_dir)
        self.assertEqual((manifest["status"], manifest["stage"]), ("success", "done"))
        self.assertEqual(len(manifest["selected"]), 1)
        self.assertEqual([entry["index"] for entry in manifest["selected"]], [VIOLATION])

    def test_image_reading_failure_marks_the_manifest_failed(self):
        facts = facts_for(root=Path(self.tmp.name))
        facts["frame_images"][0]["images"]["CAM_F0"] = "/nonexistent/frame.jpg"
        result = analyze(self.run_dir, facts)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"]["environment"]["weather"], "unknown")
        self.assertEqual(evidence.read_manifest(self.run_dir)["status"], "success")

    def test_bev_client_construction_failure_is_fatal(self):
        class BrokenClient:
            def __init__(self, component=None):
                raise RuntimeError("no api key")

        result = analyze(self.run_dir, client=BrokenClient)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "bev_check_error")
        self.assertEqual(evidence.read_manifest(self.run_dir)["status"], "failed")

    def test_empty_and_non_string_replies_degrade_only_environment(self):
        for reply in ("", "   ", None):
            with self.subTest(reply=reply):
                StubClient.answer = reply
                result = analyze(self.run_dir)
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["result"]["environment"]["weather"], "unknown")

    def test_standalone_check_creates_its_own_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(evidence, "LOG_ROOT", Path(tmp)), \
                 mock.patch("agent.llm_client.LLMClient", StubClient):
                facts = facts_for(root=Path(tmp) / "cameras")
                visual_check.check("archive", "token", facts)
            created = [path for path in Path(tmp).rglob("manifest.json")]
            self.assertEqual(len(created), 1)
            manifest = json.loads(created[0].read_text())
            self.assertEqual((manifest["status"], manifest["stage"]), ("success", "done"))
            self.assertEqual(len(manifest["frames"]), 1)

    def test_data_stage_uses_its_own_limit_not_a_worker_deadline(self):
        with mock.patch.object(real, "failure_facts",
                               return_value=facts_for(root=Path(self.tmp.name))) as call, \
             mock.patch("agent.llm_client.LLMClient", StubClient):
            real.analyze({"token": "4abb45d9ee1a5c5c", "simulation": "run/sim", "timeout": 120},
                         self.run_dir)
        self.assertEqual(call.call_args.kwargs, {})


class DirectoryTests(unittest.TestCase):
    def test_same_token_under_different_simulations_does_not_share_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = evidence.new_run_dir("experiment/a", "token", root=tmp)
            second = evidence.new_run_dir("experiment/b", "token", root=tmp)
            self.assertNotEqual(first.parent.parent, second.parent.parent)
            self.assertTrue(first.is_dir() and second.is_dir())

    def test_repeat_runs_of_the_same_scene_get_unique_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = {evidence.new_run_dir("experiment/a", "token", root=tmp) for _ in range(5)}
            self.assertEqual(len(runs), 5)

    def test_directory_name_is_filesystem_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = evidence.new_run_dir("../../etc/exp a/../b", "tok en", root=tmp)
            self.assertTrue(run_dir.is_dir())
            self.assertTrue(run_dir.is_relative_to(Path(tmp)))
            self.assertNotIn("..", run_dir.parent.parent.name)


if __name__ == "__main__":
    unittest.main()
