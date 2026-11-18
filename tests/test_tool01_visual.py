"""The visual check must describe the frames it really sends and keep them on disk."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.tool01_failure_analyzer import evidence, real, visual_check  # noqa: E402

CAMERA_TMP = tempfile.TemporaryDirectory(prefix="t01cam-")
RUN_TMP = tempfile.TemporaryDirectory(prefix="t01run-")

FACTS = {"failure": {"outcome": "collision", "interaction": "crossing"},
         "attack": {"map_context": "t_junction", "ego_action": "turn_right"}}


class StubClient:
    """Stands in for the model client; the answer is set per test."""

    answer = ""
    calls = []

    def __init__(self, component=None):
        self.component = component

    def chat(self, messages):
        StubClient.calls.append(messages)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def answer(**overrides):
    payload = {"matched": True, "notes": [], "weather": "cloudy", "extra": "白天，路面干燥"}
    payload.update(overrides)
    for name in [key for key, value in overrides.items() if value is _MISSING]:
        payload.pop(name)
    return json.dumps(payload, ensure_ascii=False)


class _Missing:
    pass


_MISSING = _Missing()
_DEFAULT = object()


def tearDownModule():
    CAMERA_TMP.cleanup()
    RUN_TMP.cleanup()


def camera_frames(count):
    """Real, valid JPEG files for all eight cameras, removed when the module ends."""
    root = Path(CAMERA_TMP.name) / f"scene-{count}" / "sensor_blobs" / "scene"
    images = {}
    for camera in visual_check.CAMERAS:
        (root / camera).mkdir(parents=True, exist_ok=True)
        images[camera] = []
        for index in range(count):
            path = root / camera / f"{index:02d}.jpg"
            Image.new("RGB", (64, 36), (index * 11 % 256, 60, 120)).save(path, format="JPEG", quality=90)
            images[camera].append(path)
    return [{"index": index, "time_s": index * 0.5,
             "images": {camera: str(images[camera][index]) for camera in visual_check.CAMERAS}}
            for index in range(count)]


def run_check(payload, count=12, violation=_DEFAULT, run_dir=None):
    StubClient.answer = payload
    StubClient.calls = []
    frames = camera_frames(count)
    facts = dict(FACTS, violation_step=7 if violation is _DEFAULT else violation, frame_images=frames)
    directory = Path(run_dir) if run_dir else Path(RUN_TMP.name) / f"run-{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=True)
    with mock.patch("agent.llm_client.LLMClient", StubClient):
        result = visual_check.check("archive", "token", facts, run_dir=directory)
    return result, directory


def last_prompt():
    content = StubClient.calls[-1][1]["content"]
    return content[0]["text"], [part for part in content if part["type"] == "image_url"]


def listed_frames(text):
    return [int(value) for value in re.findall(r"- 第 \d+ 张：帧 (\d+)，", text)]


class SampledFrameTests(unittest.TestCase):
    def test_four_images_inside_the_scene(self):
        _, directory = run_check(answer())
        text, images = last_prompt()
        self.assertEqual(listed_frames(text), [0, 6, 8, 11])
        self.assertEqual(len(images), 4)
        self.assertIn("首次违规帧为第 7 帧", text)
        self.assertIn("你会收到 4 张", text)
        self.assertIn("- 第 2 张：帧 6，3.0s", text)
        self.assertIn("首次违规前一帧", text)
        self.assertNotIn("事故前", text)
        self.assertNotIn("事故后", text)

    def test_boundary_violations_keep_the_real_frames(self):
        cases = {(0, 12): [0, 1, 11], (1, 12): [0, 2, 11], (10, 12): [0, 9, 11],
                 (11, 12): [0, 10, 11], (0, 1): [0], (1, 2): [0, 1]}
        for (violation, count), expected in cases.items():
            with self.subTest(violation=violation, count=count):
                run_check(answer(), count=count, violation=violation)
                text, images = last_prompt()
                self.assertEqual(listed_frames(text), expected)
                self.assertEqual(len(images), len(expected))
                self.assertIn(f"你会收到 {len(expected)} 张", text)

    def test_sent_images_match_the_listing(self):
        for count, violation in ((12, 7), (12, 0), (12, 11), (1, 0), (2, 1), (3, 1)):
            with self.subTest(count=count, violation=violation):
                run_check(answer(), count=count, violation=violation)
                text, images = last_prompt()
                expected = visual_check.sheet_indices(count, violation)
                self.assertEqual(listed_frames(text), expected)
                self.assertEqual(len(images), len(expected))

    def test_bad_frames_are_rejected(self):
        for violation in (-1, 12, 7.5, 0.0, True, "7", None):
            with self.subTest(violation=violation):
                with self.assertRaises(visual_check.VisualCheckError):
                    run_check(answer(), violation=violation)
    def test_bad_counts_are_rejected(self):
        for count in (0, -1, 2.5, True):
            with self.subTest(count=count):
                with self.assertRaises(visual_check.VisualCheckError):
                    visual_check.sheet_indices(count, 0)


class AcceptedAnswers(unittest.TestCase):
    def test_complete_answer_passes(self):
        result, _ = run_check(answer())
        self.assertEqual(result, {"matched": True, "notes": [], "weather": "cloudy",
                                  "extra": "白天，路面干燥"})

    def test_explicit_unknown_and_empties_pass(self):
        result, _ = run_check(answer(weather="unknown", extra="", notes=[]))
        self.assertEqual((result["weather"], result["extra"], result["notes"]), ("unknown", "", []))

    def test_weather_is_normalized(self):
        self.assertEqual(run_check(answer(weather=" Cloudy "))[0]["weather"], "cloudy")

    def test_matched_false_reaches_the_mismatch_path(self):
        result, directory = run_check(answer(matched=False, notes=["画面里自车没有右转"]))
        self.assertFalse(result["matched"])
        self.assertEqual(result["notes"], ["画面里自车没有右转"])
        self.assertEqual(evidence.read_manifest(directory)["verification"]["matched"], False)


class RejectedAnswers(unittest.TestCase):
    def assert_rejected(self, payload, *expected):
        with self.assertRaises(visual_check.VisualCheckError) as caught:
            run_check(payload)
        for text in expected:
            self.assertIn(text, str(caught.exception))
        self.assertEqual(caught.exception.stage, "vlm")

    def test_missing_fields_are_reported(self):
        for name in ("matched", "notes", "weather", "extra"):
            with self.subTest(name):
                self.assert_rejected(answer(**{name: _MISSING}), name)

    def test_non_boolean_matched_is_rejected(self):
        for value in (1, 0, "true", None):
            with self.subTest(value=value):
                self.assert_rejected(answer(matched=value), "matched")

    def test_bad_notes_are_rejected(self):
        for value in (False, None, "no notes", [1, None], [""], ["  "]):
            with self.subTest(value=value):
                self.assert_rejected(answer(notes=value), "notes")

    def test_notes_must_match_the_verification_verdict(self):
        self.assert_rejected(answer(matched=True, notes=["画面有差异"]), "empty", "true")
        self.assert_rejected(answer(matched=False, notes=[]), "explain", "false")

    def test_unexpected_fields_are_rejected(self):
        payload = json.loads(answer())
        payload["reasoning"] = "not part of the contract"
        self.assert_rejected(json.dumps(payload), "unexpected fields", "reasoning")

    def test_bad_weather_is_rejected(self):
        for value in (None, 3, "", "sunny"):
            with self.subTest(value=value):
                self.assert_rejected(answer(weather=value), "weather")

    def test_non_string_extra_is_rejected(self):
        for value in (None, 1, {"text": "白天"}):
            with self.subTest(value=value):
                self.assert_rejected(answer(extra=value), "extra")


if __name__ == "__main__":
    unittest.main()
