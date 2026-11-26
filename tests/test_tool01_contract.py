"""Unit and contract tests for the refactored Tool 01 (failure_analyzer)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from mcp_servers.worldengine_failure_analysis.adapter import service
from tools.registry import call_tool
from tools.tool01_failure_analyzer import real, visual_check
from tools.tool01_failure_analyzer.schemas import Output


def _sample_facts(tmp: Path, *, outcome="collision", nearby=None):
    bev = tmp / "src_bev.png"
    Image.new("RGB", (64, 64), (255, 255, 255)).save(bev)
    traj = tmp / "src_traj.json"
    traj.write_text(json.dumps({"EGO": []}))
    cams = {}
    for cam in visual_check.CAMERAS:
        p = tmp / f"{cam}.jpg"
        Image.new("RGB", (32, 24), (200, 200, 200)).save(p)
        cams[cam] = str(p)
    if nearby is None:
        nearby = [{
            "vehicle": "V1",
            "track": "track_bus_1",
            "label": "bus",
            "size_m": [11.87, 2.86, 3.45],
            "frames_observed": [0, 8],
            "min_distance_m": 0.0,
            "violation_distance_m": 0.0,
            "bearing_at_violation": "正前方",
            "relative_heading_at_violation_deg": 166.9,
            "kinematics": {
                "speed_start_mps": 9.3,
                "speed_violation_mps": 9.6,
                "displacement_m": 38.3,
                "yaw_change_deg": 85.1,
                "tail_yaw_deg": 47.9,
                "is_stationary": False,
            },
        }]
    return {
        "status": "ok",
        "token": "scene-4abb45d9ee1a5c5c",
        "violation_step": 0,
        "map_location": "us-pa-pittsburgh-hazelwood",
        "outcome": outcome,
        "ego_kinematics": {
            "speed_start_mps": 3.5,
            "speed_violation_mps": 3.4,
            "displacement_m": 15.4,
            "yaw_change_deg": -54.7,
            "tail_yaw_deg": -4.0,
            "is_stationary": False,
        },
        "nearby_vehicles": nearby,
        "bev_image": str(bev),
        "trajectories_path": str(traj),
        "environment": {"time": "2021-09-16T20:16:41+00:00", "weather": None, "extra": None},
        "frame_images": [{"index": 0, "time_s": 0.0, "images": cams}],
        "archive": str(tmp),
        "simulation_record": "exp/batch",
    }


class _FakeClient:
    replies: list[str] = []

    def __init__(self, component=None):
        self.component = component

    def chat(self, messages):
        return self.replies.pop(0)


class Tool01ContractTests(unittest.TestCase):
    def test_mock_matches_output_schema(self):
        res = call_tool("failure_analyzer", {"token": "tok_1", "simulation": "sim_1"}, mode="mock")
        self.assertEqual(res["status"], "success")
        Output.model_validate(res["result"])
        self.assertIn("kinematics", res["result"])
        self.assertIn("bev_image", res["result"])

    def test_yaw_text_uses_explicit_directions(self):
        self.assertEqual(visual_check._yaw_text(85.1), "左转85.1°")
        self.assertEqual(visual_check._yaw_text(-54.7), "右转54.7°")
        self.assertEqual(visual_check._yaw_text(-4.0), "近直行（偏转4.0°）")

    def test_select_prompt_vehicles_single_target_for_collision(self):
        with tempfile.TemporaryDirectory() as td:
            v1 = {"track": "a", "violation_distance_m": 0.0, "kinematics": {"is_stationary": True}}
            v2 = {"track": "b", "violation_distance_m": 3.6, "kinematics": {"is_stationary": True}}
            facts = {"outcome": "collision", "nearby_vehicles": [v2, v1]}
            selected = visual_check.select_prompt_vehicles(facts)
            self.assertEqual([v["track"] for v in selected], ["a"])

            dep_facts = {"outcome": "road_departure", "nearby_vehicles": [v2, v1]}
            dep_selected = visual_check.select_prompt_vehicles(dep_facts)
            self.assertEqual([v["track"] for v in dep_selected], ["a", "b"])

    def test_bev_rejects_inconsistent_or_invalid_replies(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bad_cases = [
                ("collision", None, json.dumps({
                    "map_context": "curve", "failure_driver": "departure_ego_only",
                    "ego_behavior": "a", "other_behavior": "", "summary": "s",
                })),
                ("road_departure", [], json.dumps({
                    "map_context": "curve", "failure_driver": "departure_by_other",
                    "ego_behavior": "a", "other_behavior": "b", "summary": "s",
                })),
                ("road_departure", [], json.dumps({
                    "map_context": "curve", "failure_driver": "departure_ego_only",
                    "ego_behavior": "a", "other_behavior": "不应有他车", "summary": "s",
                })),
                ("collision", None, json.dumps({
                    "map_context": "bad_road", "failure_driver": "collision_with_vehicle",
                    "ego_behavior": "a", "other_behavior": "b", "summary": "s",
                })),
            ]
            for idx, (out, near, reply) in enumerate(bad_cases):
                facts = _sample_facts(tmp, outcome=out, nearby=near)
                _FakeClient.replies = [reply]
                with mock.patch("agent.llm_client.LLMClient", _FakeClient):
                    with self.assertRaises(visual_check.VisualCheckError):
                        visual_check.check_bev(facts, tmp / f"run_{idx}")

    def test_analyze_collision_and_ego_only_departure(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            # 1. Collision with bus
            facts = _sample_facts(tmp, outcome="collision")
            _FakeClient.replies = [
                json.dumps({
                    "map_context": "curve",
                    "failure_driver": "collision_with_vehicle",
                    "ego_behavior": "弯道低速行驶，末段近直行",
                    "other_behavior": "正前方对向大型车左转迎面相撞",
                    "summary": "弯道上自车与对向左转大型车在第0帧相撞。",
                }),
                json.dumps({"weather": "clear", "extra": "白天晴朗干燥", "notes": []}),
            ]
            with mock.patch.object(real, "failure_facts", return_value=facts), \
                 mock.patch("agent.llm_client.LLMClient", _FakeClient):
                res = real.analyze({"token": "4abb", "simulation": "exp/batch"}, tmp / "run_col")
            self.assertEqual(res["status"], "success")
            validated = Output.model_validate(res["result"])
            self.assertEqual(validated.failure["target_track"], "track_bus_1")
            self.assertEqual(validated.failure["target_class"], "bus")
            self.assertIsNotNone(validated.kinematics["other"])
            self.assertTrue(Path(validated.bev_image).is_file())
            self.assertTrue(Path(validated.kinematics["trajectory_ref"]).is_file())

            # 2. Ego-only road departure (0 nearby vehicles)
            facts_dep = _sample_facts(tmp, outcome="road_departure", nearby=[])
            _FakeClient.replies = [
                json.dumps({
                    "map_context": "curve",
                    "failure_driver": "departure_ego_only",
                    "ego_behavior": "弯道减速行驶末段偏出边界",
                    "other_behavior": "",
                    "summary": "弯道上自车偏离压出道路边界。",
                }),
                json.dumps({"weather": "cloudy", "extra": "阴天路面干燥", "notes": []}),
            ]
            with mock.patch.object(real, "failure_facts", return_value=facts_dep), \
                 mock.patch("agent.llm_client.LLMClient", _FakeClient):
                res_dep = real.analyze({"token": "4a91", "simulation": "exp/batch"}, tmp / "run_dep")
            self.assertEqual(res_dep["status"], "success")
            v_dep = Output.model_validate(res_dep["result"])
            self.assertIsNone(v_dep.failure["target_track"])
            self.assertEqual(v_dep.failure["target_class"], "none")
            self.assertEqual(v_dep.failure["other_behavior"], "")
            self.assertIsNone(v_dep.kinematics["other"])

    def test_adapter_rejects_incomplete_facts(self):
        for missing_key in service.REQUIRED_FACT_KEYS:
            payload = {k: "x" for k in service.REQUIRED_FACT_KEYS}
            payload["status"] = "ok"
            payload.pop(missing_key)
            completed = mock.Mock(returncode=0, stdout=json.dumps(payload), stderr="")
            with mock.patch.object(service, "resolve_simulation", return_value=Path("/tmp/sim")), \
                 mock.patch.object(service.subprocess, "run", return_value=completed):
                with self.assertRaises(service.WfaError) as ctx:
                    service.failure_facts("tok", "exp/batch")
                self.assertIn(missing_key, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
