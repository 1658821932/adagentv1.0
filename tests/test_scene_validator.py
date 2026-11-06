import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.tool10_scene_validator.real import run


class SceneValidatorTests(unittest.TestCase):
    def test_known_reconstruction_artifacts_do_not_block_edit_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "rendered.mp4"
            video.touch()
            evaluation = {
                "requirement_consistency": True,
                "visual_valid": True,
                "geometry_valid": True,
                "physics_valid": True,
                "traffic_logic_valid": True,
                "issues": [],
                "ignored_reconstruction_artifacts": ["画面中央的黑色横向空洞"],
            }
            with patch("tools.tool10_scene_validator.real._evaluate_visual", return_value=evaluation):
                result = run({"rendered_video": str(video), "scene_requirement": "前方有救护车"})["result"]
            self.assertTrue(result["passed"])
            self.assertEqual(result["ignored_reconstruction_artifacts"], evaluation["ignored_reconstruction_artifacts"])

    def test_edit_specific_issue_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "rendered.mp4"
            video.touch()
            evaluation = {
                "requirement_consistency": False,
                "visual_valid": False,
                "geometry_valid": False,
                "physics_valid": False,
                "traffic_logic_valid": False,
                "issues": ["救护车未清晰出现"],
                "ignored_reconstruction_artifacts": ["右侧山体破碎"],
            }
            with patch("tools.tool10_scene_validator.real._evaluate_visual", return_value=evaluation):
                result = run({"rendered_video": str(video), "scene_requirement": "前方有救护车"})["result"]
            self.assertFalse(result["passed"])
            self.assertEqual(result["issues"], ["救护车未清晰出现"])

    def test_bad_grounding_fails_even_when_requirement_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "rendered.mp4"
            video.touch()
            evaluation = {
                "requirement_consistency": True,
                "visual_valid": True,
                "geometry_valid": False,
                "physics_valid": True,
                "traffic_logic_valid": True,
                "issues": [],
            }
            with patch("tools.tool10_scene_validator.real._evaluate_visual", return_value=evaluation):
                result = run({"rendered_video": str(video), "scene_requirement": "前车切入"})["result"]
            self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()



def test_vlm_template_uses_runtime_schema_and_keeps_context_separate(tmp_path):
    import json
    from tools.tool10_scene_validator import real
    frames=[]
    for i in range(8):
        p=tmp_path/f"frame_{i}.jpg";p.write_bytes(b"test-frame");frames.append(str(p))
    evaluation={name:True for name in ("requirement_consistency","geometry_valid","visual_valid","physics_valid","traffic_logic_valid")}
    settings=real.VLMSettings(base_url="https://offline.invalid/v1",model="offline",timeout=10,api_key="test-only")
    response={"choices":[{"message":{"content":json.dumps(evaluation)}}]}
    with patch.object(real,"vlm_settings",return_value=settings),patch.object(real,"_chat_completion",return_value=response) as chat:
        result=real._evaluate_visual({"frame_paths":frames,"ignored_reconstruction_artifacts":["known source hole"]},tmp_path/"render.mp4","lead vehicle slows down")
    content=chat.call_args.args[2]["messages"][0]["content"]
    prompt=content[0]["text"]
    assert not any(token in prompt for token in ("{response_schema}","{ignored_artifacts}","{requirement}"))
    assert "lead vehicle slows down" in prompt and "known source hole" in prompt
    schema_text=prompt.split("# 输出",1)[1].split("# 最初场景需求",1)[0].split("\n",2)[-1].strip()
    schema=json.loads(schema_text)
    assert set(schema["required"])==set(evaluation)
    assert "model" not in schema["properties"] and "sampled_frames" not in schema["properties"]
    assert len(content)==9 and result["sampled_frames"]==8
