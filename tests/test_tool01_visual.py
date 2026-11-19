"""The visual check only supplements weather and environment text."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.tool01_failure_analyzer import visual_check  # noqa: E402

CAMERA_TMP = tempfile.TemporaryDirectory(prefix="t01cam-")
RUN_TMP = tempfile.TemporaryDirectory(prefix="t01run-")
FACTS = {
    "failure": {"outcome": "collision", "interaction": "crossing"},
    "attack": {"map_context": "t_junction", "ego_action": "turn_right",
               "ego_lane_keeping": "keep", "other_origin": "right_arm",
               "other_action": "turn_left", "other_lane_keeping": "keep"},
}


class StubClient:
    answer = ""
    calls = []

    def __init__(self, component=None):
        self.component = component

    def chat(self, messages):
        StubClient.calls.append(messages)
        return self.answer


class _Missing:
    pass


MISSING = _Missing()


def answer(**overrides):
    payload = {"weather": "cloudy", "extra": "白天，路面干燥", "notes": []}
    payload.update(overrides)
    for key in [key for key, value in overrides.items() if value is MISSING]:
        payload.pop(key)
    return json.dumps(payload, ensure_ascii=False)


def camera_frames(count):
    root = Path(CAMERA_TMP.name) / f"scene-{count}" / "sensor_blobs" / "scene"
    rows = []
    for index in range(count):
        images = {}
        for camera in visual_check.CAMERAS:
            folder = root / camera
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{index:02d}.jpg"
            Image.new("RGB", (64, 36), (index * 11 % 256, 60, 120)).save(path, format="JPEG")
            images[camera] = str(path)
        rows.append({"index": index, "time_s": index * 0.5, "images": images})
    return rows


def run_check(payload, count=12, violation=7):
    StubClient.answer = payload
    StubClient.calls = []
    facts = dict(FACTS, violation_step=violation, frame_images=camera_frames(count))
    directory = Path(RUN_TMP.name) / f"run-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    with mock.patch("agent.llm_client.LLMClient", StubClient):
        result = visual_check.check("archive", "token", facts, run_dir=directory)
    return result


def last_prompt():
    content = StubClient.calls[-1][1]["content"]
    return content[0]["text"], [part for part in content if part["type"] == "image_url"]


class VisualEnvironmentTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        CAMERA_TMP.cleanup()
        RUN_TMP.cleanup()

    def test_only_violation_frame_is_sent_for_environment(self):
        run_check(answer())
        text, images = last_prompt()
        self.assertEqual(len(images), 1)
        self.assertIn("首次违规帧为第 7 帧", text)
        self.assertIn("你会收到 1 张", text)

    def test_valid_environment_answer_has_no_corrections(self):
        result = run_check(answer())
        self.assertEqual(result["sample_frames"], [7])
        self.assertEqual(result["corrections"], [])
        self.assertEqual(result["weather"], "cloudy")
        self.assertEqual(result["extra"], "白天，路面干燥")
        self.assertNotIn("decision", result)

    def test_weather_is_normalized_and_unknown_is_allowed(self):
        self.assertEqual(run_check(answer(weather=" Cloudy "))["weather"], "cloudy")
        result = run_check(answer(weather="unknown", extra=""))
        self.assertEqual((result["weather"], result["extra"]), ("unknown", ""))

    def test_bad_violation_frames_are_rejected(self):
        for violation in (-1, 12, 7.5, 0.0, True, "7", None):
            with self.subTest(violation=violation):
                with self.assertRaises(visual_check.VisualCheckError):
                    run_check(answer(), violation=violation)

    def test_bad_counts_are_rejected(self):
        for count in (0, -1, 2.5, True):
            with self.subTest(count=count):
                with self.assertRaises(visual_check.VisualCheckError):
                    visual_check.sheet_indices(count, 0)

    def test_missing_environment_fields_are_rejected(self):
        for name in ("weather", "extra", "notes"):
            with self.subTest(name=name):
                with self.assertRaises(visual_check.VisualCheckError):
                    run_check(answer(**{name: MISSING}))

    def test_key_data_fields_are_rejected(self):
        for name in ("map_context", "ego_action", "other_origin", "other_action", "interaction"):
            with self.subTest(name=name):
                payload = json.loads(answer())
                payload[name] = "x"
                with self.assertRaises(visual_check.VisualCheckError):
                    run_check(json.dumps(payload))

    def test_invalid_notes_weather_and_extra_are_rejected(self):
        for payload in (answer(notes=False), answer(notes=[1]), answer(notes=[""]),
                        answer(weather=None), answer(weather="sunny"),
                        answer(extra=None), answer(extra=3)):
            with self.subTest(payload=payload):
                with self.assertRaises(visual_check.VisualCheckError):
                    run_check(payload)


if __name__ == "__main__":
    unittest.main()
