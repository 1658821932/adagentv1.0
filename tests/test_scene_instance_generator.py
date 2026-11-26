"""Unit and contract tests for refactored Tool 02 (scene_instance_generator)."""
from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tools.tool01_failure_analyzer import mock as tool01_mock
from tools.tool02_scene_instance_generator import mock as tool02_mock, real as tool02_real
from tools.tool02_scene_instance_generator.schemas import Input, Output


class _StubClient:
    def __init__(self, reply: dict):
        self.reply = reply
        self.model = "stub-model"
        self.last_usage = None
        self.seen_messages = []

    def chat(self, messages):
        self.seen_messages.append(messages)
        return json.dumps(self.reply, ensure_ascii=False)


def _ego_only_tool01() -> dict:
    base = deepcopy(tool01_mock.run({"token": "dep_tok", "simulation": "exp/batch"}))
    base["result"]["failure"] = {
        "outcome": "road_departure",
        "failure_driver": "departure_ego_only",
        "target_track": None,
        "target_class": "none",
        "map_context": "curve",
        "ego_behavior": "弯道减速末段压出外侧边界",
        "other_behavior": "",
        "summary": "弯道上自车末段压出外侧边界。",
    }
    base["result"]["kinematics"]["other"] = None
    return base


class Tool02ContractTests(unittest.TestCase):
    def test_mock_vehicle_driven_prioritizes_add_vehicle_and_injects_size(self):
        t1 = tool01_mock.run({"token": "bus_tok", "simulation": "exp/batch"})
        res = tool02_mock.run({"failure_analysis": t1})
        self.assertEqual(res["status"], "success")
        out = Output.model_validate(res["result"])
        self.assertEqual(out.failure_context.map_context, "curve")
        self.assertEqual(out.failure_context.failure_driver, "collision_with_vehicle")
        self.assertIsNotNone(out.candidates[0].asset_requirement)
        self.assertEqual(out.candidates[0].asset_requirement.vehicle_class, "bus")
        self.assertEqual(out.candidates[0].asset_requirement.size_m, [11.87, 2.86, 3.45])
        self.assertIsNone(out.candidates[2].asset_requirement)
        self.assertEqual(out.kinematics["trajectory_ref"], t1["result"]["kinematics"]["trajectory_ref"])

    def test_mock_ego_only_has_null_assets_and_attack_strategies(self):
        t1 = _ego_only_tool01()
        res = tool02_mock.run({"failure_analysis": t1})
        self.assertEqual(res["status"], "success")
        out = Output.model_validate(res["result"])
        self.assertEqual(out.failure_context.failure_driver, "departure_ego_only")
        for cand in out.candidates:
            self.assertIsNone(cand.asset_requirement)
            self.assertIsNone(cand.attack_edit_strategy)

    def test_payload_strips_local_paths_and_source(self):
        t1 = tool01_mock.run({"token": "bus_tok", "simulation": "exp/batch"})
        data = Input.model_validate({"failure_analysis": t1})
        payload = tool02_real._payload(data)["failure_analysis"]
        self.assertNotIn("source", payload)
        self.assertNotIn("trajectory_ref", payload["kinematics"])
        self.assertNotIn("target_track", payload["failure"])

    def test_rejects_cand1_without_asset_for_vehicle_failure(self):
        t1 = tool01_mock.run({"token": "bus_tok", "simulation": "exp/batch"})
        bad = deepcopy(tool02_mock.RESULT)
        bad["candidates"][0]["asset_requirement"] = None
        stub = _StubClient(bad)
        res = tool02_real.execute({"failure_analysis": t1, "_llm_client": stub})
        self.assertEqual(res["status"], "error")
        self.assertIn("cand_1", json.dumps(res["error"], ensure_ascii=False))

    def test_rejects_banned_terms_in_retrieval_query(self):
        t1 = tool01_mock.run({"token": "bus_tok", "simulation": "exp/batch"})
        bad = deepcopy(tool02_mock.RESULT)
        bad["candidates"][0]["retrieval_query"] = "白天弯道发生碰撞与冲突"
        stub = _StubClient(bad)
        res = tool02_real.execute({"failure_analysis": t1, "_llm_client": stub})
        self.assertEqual(res["status"], "error")
        self.assertIn("retrieval_query", json.dumps(res["error"], ensure_ascii=False))

    def test_accepts_result_ref_file(self):
        t1 = tool01_mock.run({"token": "bus_tok", "simulation": "exp/batch"})
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "tool01.json"
            ref.write_text(json.dumps(t1, ensure_ascii=False))
            res = tool02_mock.run({"failure_analysis": str(ref)})
            self.assertEqual(res["status"], "success")


if __name__ == "__main__":
    unittest.main()
