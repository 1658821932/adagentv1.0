"""Mock reports the requested mode and never produces real training artifacts."""
def run(input_data: dict) -> dict:
    mode = input_data.get("training_mode", "ilft")
    return {"status": "success", "tool": "policy_trainer", "result": {
        "training_mode": mode, "mock": True,
        "log_path": "outputs/mock/training.log",
        "model_id": "mock_policy_v2",
        "checkpoint": "outputs/mock/model_v2.ckpt"}}
