"""Mode dispatch and rejection tests: do not launch GPU training."""
import unittest
from pydantic import ValidationError
from tools.tool15_policy_trainer.schemas import Input, Output
from tools.tool15_policy_trainer.mock import run

class TrainingModeTests(unittest.TestCase):
    def test_agent_description_exposes_both_modes(self):
        from tools.tool15_policy_trainer.description import DESCRIPTION
        modes = DESCRIPTION["input_schema"]["properties"]["training_mode"]["enum"]
        self.assertEqual(set(modes), {"ilft", "rlft"})

    def test_original_il_payload_still_works(self):
        obj = Input.model_validate(dict(base_checkpoint="/weights/base.pth",
            training_data=dict(ann_file="/data/infos.pkl", token_file="/data/tokens.yaml")))
        self.assertEqual(obj.training_mode, "ilft")

    def test_rl_requires_rewards_and_vocabulary_inputs(self):
        with self.assertRaises(ValidationError):
            Input.model_validate(dict(training_mode="rlft", base_checkpoint="/base.pth",
                training_data=dict(token_file="/tokens.yaml")))

    def test_il_rejects_rl_inputs(self):
        with self.assertRaises(ValidationError):
            Input.model_validate(dict(training_mode="ilft", base_checkpoint="/base.pth",
                training_data=dict(ann_file="/ann.pkl", token_file="/tokens.yaml",
                    rollout_dir="/rollout", vocab_path="/vocab.npy")))

    def test_rl_rejects_annotation_only_input(self):
        with self.assertRaises(ValidationError):
            Input.model_validate(dict(training_mode="rlft", base_checkpoint="/base.pth",
                training_data=dict(ann_file="/ann.pkl", token_file="/tokens.yaml")))

    def test_modes_preserved_in_mock_output(self):
        for mode in ["ilft", "rlft"]:
            result = run({"training_mode": mode})["result"]
            self.assertEqual(Output.model_validate(result).training_mode, mode)
            self.assertTrue(result["mock"])

    def test_valid_rl_and_unknown_mode(self):
        payload = dict(training_mode="rlft", base_checkpoint="/base.pth",
            training_data=dict(rollout_dir="/rollout", token_file="/tokens.yaml", vocab_path="/vocab.npy"))
        self.assertEqual(Input.model_validate(payload).training_mode, "rlft")
        with self.assertRaises(ValidationError):
            Input.model_validate({**payload, "training_mode": "unknown"})

if __name__ == "__main__":
    unittest.main()
