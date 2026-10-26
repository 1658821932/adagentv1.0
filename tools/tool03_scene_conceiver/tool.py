"""Local mock implementation of the scene conceiver."""


def run(input_data: dict) -> dict:
    """Return a deterministic mock scene conception."""
    return {
        "status": "success",
        "tool": "scene_conceiver",
        "result": {
            "base_video_requirement": "Forward-facing urban driving video with two same-lane vehicles.",
            "required_conditions": ["daytime", "dry road", "straight lane"],
            "simple_edits": ["increase lead vehicle braking intensity"],
            "expected_effect": "Produce a realistic lead-vehicle braking conflict.",
        },
    }
