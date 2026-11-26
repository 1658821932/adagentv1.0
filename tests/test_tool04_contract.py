"""Contract and unit tests for refactored Dual-BEV Tool 04 (edit_planner)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from tools.registry import call_tool
from tools.tool01_failure_analyzer import mock as tool01_mock
from tools.tool02_scene_instance_generator import mock as tool02_mock
from tools.tool04_edit_planner import real as tool04_real
from tools.tool04_edit_planner.schemas import Output


class _StubClient:
    def __init__(self, reply: dict):
        self.reply = reply
        self.model = "stub-vlm"

    def chat(self, messages):
        return json.dumps(self.reply, ensure_ascii=False)


def _scene_ref(tmp: Path) -> dict:
    frames = tmp / "frames.json"
    cams = tmp / "cameras.json"
    pkl = tmp / "all_scenarios.pkl"
    manifest = tmp / "scene_binding.json"
    frames.write_text("[]")
    cams.write_text("{}")
    pkl.write_bytes(b"")
    manifest.write_text("{}")
    return {
        "scene_id": "scene_01",
        "scene_token": "tok_01",
        "sample_root": str(tmp),
        "dataset": "navtrain_ep_per1",
        "matched_camera": "CAM_F0",
        "frame_count": 24,
        "fps": 2.0,
        "scenario_pkl": str(pkl),
        "frames_path": str(frames),
        "cameras_path": str(cams),
        "manifest_path": str(manifest),
        "images_complete": True,
        "cameras": {},
    }


class Tool04ContractTests(unittest.TestCase):
    def test_mock_tool04_builds_resized_add_vehicle_handoff(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            t1 = tool01_mock.run({"token": "bus_tok", "simulation": "exp/batch"})
            t2 = tool02_mock.run({"failure_analysis": t1})
            t3 = {"status": "success", "tool": "video_retriever",
                  "result": {"scene_ref": _scene_ref(tmp), "candidate_id": "cand_1", "mode": "real"}}
            res = call_tool("edit_planner", {
                "scenario_result": t2, "candidate_id": "cand_1", "retrieval_result": t3
            }, mode="mock")
            self.assertEqual(res["status"], "success")
            out = Output.model_validate(res["result"])
            self.assertEqual(out.suitability, "ready")
            self.assertEqual(out.attack_input["operation"], "add_vehicle")
            self.assertEqual(out.attack_input["vehicle"]["size_m"], [4.432, 2.033, 1.637])
            self.assertIsNone(out.attack_input["constraints"]["target_window_s"])

    def test_real_dual_bev_ready_and_unsuitable(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src_bev = tmp / "src_bev.png"
            ret_bev = tmp / "ret_bev.png"
            Image.new("RGB", (64, 64), (255, 255, 255)).save(src_bev)
            Image.new("RGB", (64, 64), (240, 240, 240)).save(ret_bev)
            t1 = tool01_mock.run({"token": "bus_tok", "simulation": "exp/batch"})
            t1["result"]["bev_image"] = str(src_bev)
            t2 = tool02_mock.run({"failure_analysis": t1})
            t3 = {"status": "success", "tool": "video_retriever",
                  "result": {"scene_ref": _scene_ref(tmp), "candidate_id": "cand_1", "mode": "real"}}
            rendered = {
                "ok": True,
                "frame_count": 24,
                "eval_frames": 16,
                "eval_duration_s": 7.5,
                "ego_kinematics": {
                    "speed_start_mps": 8.0, "speed_violation_mps": 2.0,
                    "displacement_m": 44.0, "yaw_change_deg": -67.0, "tail_yaw_deg": -7.0, "is_stationary": False,
                },
                "nearby_vehicles": [],
                "behind_only_count": 8,
                "bev_image": str(ret_bev),
                "trajectories_path": str(tmp / "ret_traj.json"),
                "ego_bev_xy": [[2.0 * i, 0.0] for i in range(16)],
                "placeable_lanes": [
                    [[55.0, 3.5], [28.0, 3.5]],
                    [[28.0, 0.0], [55.0, 0.0]],
                ],
            }
            pose = {"ego2global": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
            (tmp / "frames.json").write_text(json.dumps([pose]))

            # Ready case
            ready_reply = {
                "suitability": "ready",
                "checks": {"road": "pass", "lanes": "pass", "space": "pass"},
                "reasons": ["右转弯道匹配且前方空旷"],
                "target_window_s": [3.0, 5.5],
                "target_vehicle_id": None,
                "adapted_plan": {
                    "conflict_role": "oncoming",
                    "attack_edit_strategy": "在对向弯道插入11.9m公交车左转下行",
                    "attack_plans": [
                        {"start_xy": [x, 3.5], "rationale": "沿对向车道驶向自车"}
                        for x in (36.0, 42.0, 48.0)
                    ],
                    "ego_input_edit_strategy": "沿原采集轨迹下调入弯速度",
                },
                "retrieval_suggestion": None,
            }
            with mock.patch.object(tool04_real, "_render_retrieved_bev", return_value=rendered), \
                 mock.patch.object(tool04_real, "RECORDS_ROOT", tmp / "rec"):
                res = tool04_real.run({
                    "scenario_result": t2, "candidate_id": "cand_1", "retrieval_result": t3,
                    "_llm_client": _StubClient(ready_reply),
                })
            self.assertEqual(res["status"], "success")
            out = Output.model_validate(res["result"])
            self.assertEqual(out.suitability, "ready")
            self.assertEqual(out.attack_input["operation"], "add_vehicle")
            self.assertIsNone(out.attack_input["vehicle"])
            self.assertEqual(out.attack_input["vehicle_catalog_class"], "any")
            self.assertIsNone(out.attack_input["constraints"]["target_window_s"])

            # A valid point, a receding lane, and a point between opposite flows.
            wrong = json.loads(json.dumps(ready_reply))
            starts = wrong["adapted_plan"]["attack_plans"]
            starts[0]["start_xy"] = [36.0, 0.0]
            starts[1]["start_xy"] = [42.0, 1.75]
            issues, ok = tool04_real._check_attack_plans_geometry(
                tmp / "frames.json", starts, rendered, "oncoming",
            )
            self.assertEqual(ok, ["S3=[48.0, 3.5]"])
            self.assertIn("远离自车", issues[0])
            self.assertIn("两条反向车道", issues[1])
            self.assertEqual(
                tool04_real._check_attack_plans_geometry(
                    tmp / "frames.json", starts[:1], rendered, "lead_vehicle",
                )[0], [],
            )

            # Recheck the final correction before handing any coordinates to Tool 05.
            for final_reply, expected in [
                ({"attack_plans": starts}, "needs_information"),
                ({"attack_plans": ready_reply["adapted_plan"]["attack_plans"]}, "ready"),
            ]:
                with mock.patch.object(tool04_real, "_render_retrieved_bev", return_value=rendered), \
                     mock.patch.object(tool04_real, "_render_self_check_overlay", return_value=ret_bev), \
                     mock.patch.object(tool04_real, "RECORDS_ROOT", tmp / "rec"):
                    client = _StubClient(wrong)
                    client.chat = mock.Mock(side_effect=[
                        json.dumps(wrong),
                        json.dumps({"attack_plans": starts}),
                        json.dumps(final_reply),
                    ])
                    res = tool04_real.run({
                        "scenario_result": t2, "candidate_id": "cand_1", "retrieval_result": t3,
                        "_llm_client": client,
                    })
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["result"]["suitability"], expected)
                self.assertEqual(client.chat.call_count, 3)
                self.assertEqual(res["result"]["attack_input"] is not None, expected == "ready")
                repair_prompt = client.chat.call_args_list[1].args[0][1]["content"][0]["text"]
                self.assertIn("S3=[48.0, 3.5]", repair_prompt.split("以下点位几何已合格")[1])

            # Unsuitable case
            bad_reply = {
                "suitability": "unsuitable",
                "checks": {"road": "fail", "lanes": "pass", "space": "fail"},
                "reasons": ["弯道方向相反且对向被占"],
                "target_window_s": None,
                "target_vehicle_id": None,
                "adapted_plan": None,
                "retrieval_suggestion": "白天双向右转弯道，对向车道空旷",
            }
            with mock.patch.object(tool04_real, "_render_retrieved_bev", return_value=rendered), \
                 mock.patch.object(tool04_real, "RECORDS_ROOT", tmp / "rec"):
                res_bad = tool04_real.run({
                    "scenario_result": t2, "candidate_id": "cand_1", "retrieval_result": t3,
                    "_llm_client": _StubClient(bad_reply),
                })
            self.assertEqual(res_bad["status"], "success")
            out_bad = Output.model_validate(res_bad["result"])
            self.assertEqual(out_bad.suitability, "unsuitable")
            self.assertIsNone(out_bad.attack_input)
            self.assertIsNotNone(out_bad.retrieval_suggestion)


if __name__ == "__main__":
    unittest.main()
