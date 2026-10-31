"""Verify rollout export contracts without starting a simulator."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch
from mcp_servers.worldengine.adapter import client

class RolloutExportTests(unittest.TestCase):
    def invoke(self, root, dense):
        result = root / "simulation"
        (result / "WE_output/openscene_format").mkdir(parents=True, exist_ok=True)
        payload = {"status":"success","result":{"result_dir":str(result),"reports":[{"succeeded":True}]}}
        args = dict(config_path="config.py", checkpoint_path="base.pth", model_name="test",
                    data_type="sample", max_frame_per_scene=12, with_dense_reward_manager=dense)
        with patch.dict(os.environ, {"ADAGENT_RUNTIME_ROOT":str(root / "runtime")}), \
             patch.object(client, "AGENT_PYTHON", Path(sys.executable)), \
             patch.object(client, "run_logged", return_value=CompletedProcess([], 0, json.dumps(payload), "")):
            return client.run_policy(args)

    def test_il_collection_does_not_require_rewards(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.invoke(Path(d), False)
            self.assertEqual(result["policy_result"], "completed")
            self.assertNotIn("candidate_rewards_present", result)

    def test_rl_collection_rejects_missing_rewards(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "without candidate reward"):
                self.invoke(Path(d), True)

    def test_actual_plan_indices_are_exported(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rewards = root / "simulation/WE_output/openscene_format/pdms_pkl"
            rewards.mkdir(parents=True)
            # Presence is checked here; numeric rewards are checked by the trainer preflight.
            (rewards / "sample_step_3_scores.pkl").write_bytes(b"")
            plans = root / "simulation/plan_traj"
            plans.mkdir()
            text = "prefix,step,plan_idx\nsample,4,73\n"
            (plans / "plan_idx.csv").write_text(text)
            result = self.invoke(root, True)
            self.assertTrue(result["candidate_rewards_present"])
            self.assertEqual((rewards / "plan_idx.csv").read_text(), text)

if __name__ == "__main__":
    unittest.main()
