"""Local mock implementation of the scene requirement generator."""


def run(input_data: dict) -> dict:
    """Return deterministic mock scene requirements."""
    return {
        "status": "success",
        "tool": "scene_requirement_generator",
        "result": {
            "scene_requirement": "Create a lead-vehicle sudden-braking driving scene.",
            "required_conditions": ["urban road", "same-lane lead vehicle"],
            "required_actors": ["ego vehicle", "lead vehicle"],
            "risk_factors": ["short headway", "lead vehicle deceleration"],
            "target_effect": "Trigger a braking-response challenge for the ego policy.",
        },
    }
