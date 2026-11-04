"""Synthetic PDM filtering fixture; no scores are read or computed."""
from pathlib import Path


def run(input_data: dict) -> dict:
    """Return mock filtering output without writing to a database."""
    return {
        "status": "success",
        "tool": "failure_case_filter",
        "meta": {"mode": "mock"},
        "result": {
            "is_failure_case": False,
            "pdm_scores_file": str(Path(input_data["result_dir"]) / "all_scenes_pdm_averages_NR.csv"),
            "scene_count": 1,
            "scene_results": [{"scene_id": input_data.get("scene_token") or "mock_scene", "pdm_score": 1.0}],
            "failure_cases": [],
        },
    }
