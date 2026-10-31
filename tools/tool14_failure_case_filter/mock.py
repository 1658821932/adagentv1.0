"""Local mock implementation of failure-case filtering."""


def run(input_data: dict) -> dict:
    """Return mock filtering output without writing to a database."""
    return {
        "status": "success",
        "tool": "failure_case_filter",
        "result": {
            "is_failure_case": False,
            "collision": False,
            "scene_count": 1,
            "scene_results": [{"scene_id": input_data.get("scene_token") or "mock_scene", "collision": False}],
            "failure_cases": [],
        },
    }
