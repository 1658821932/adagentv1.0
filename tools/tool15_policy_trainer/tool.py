"""Local mock implementation of policy training."""


def run(input_data: dict) -> dict:
    """Return mock model metadata without training or saving a checkpoint."""
    return {
        "status": "success",
        "tool": "policy_trainer",
        "result": {
            "model_id": "mock_policy_v2",
            "checkpoint": "outputs/mock/model_v2.ckpt",
        },
    }
