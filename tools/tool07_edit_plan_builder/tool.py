"""Local mock implementation of the Smart Edit configuration generator."""


def run(input_data: dict) -> dict:
    """Return a deterministic mock edit plan."""
    return {
        "status": "success",
        "tool": "edit_plan_builder",
        "result": {
            "edit_plan": "Adjust the lead vehicle trajectory to introduce sudden braking.",
            "target_instances": ["vehicle_01"],
            "operations": [
                {"operation": "modify_trajectory", "target": "vehicle_01"},
                {"operation": "set_deceleration", "target": "vehicle_01", "value": -5.0},
            ],
        },
    }
