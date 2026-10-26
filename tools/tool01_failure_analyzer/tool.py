"""Local mock implementation of the failure analyzer."""


def run(input_data: dict) -> dict:
    """Return a deterministic mock failure analysis."""
    return {
        "status": "success",
        "tool": "failure_analyzer",
        "result": {
            "failure_analysis": {
                "failure_type": ["foreground"],
                "failure_reason": "insufficient braking response",
                "key_scene_factors": ["lead vehicle deceleration", "short headway"],
                "risk_factors": ["collision"],
                "supporting_case_ids": ["mock-failure-001"],
                "uncovered_case_ids": [],
            },
        },
    }
