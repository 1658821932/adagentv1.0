"""Local mock implementation of policy execution and data collection."""


def run(input_data: dict) -> dict:
    """Return a mock policy execution result without loading a model."""
    return {
        "status": "success",
        "tool": "policy_runner",
        "result": {
            "result_dir": "outputs/mock/simulation",
            "reports": [[{"succeeded":True}]],
            "trajectory": [],
            "collision": False,
            "ttc": 3.5,
            "policy_result": "mock_success",
        },
    }
