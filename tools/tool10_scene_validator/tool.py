"""Local mock implementation of the scene validator."""


def run(input_data: dict) -> dict:
    """Return a passing validation result unless explicitly forced to fail."""
    forced_failure = input_data.get("mock_force_fail", False)
    validation = {
        "passed": not forced_failure,
        "requirement_consistency": True,
        "geometry_valid": not forced_failure,
        "physics_valid": True,
        "traffic_logic_valid": True,
        "visual_valid": True,
        "issues": ["mock geometry conflict"] if forced_failure else [],
    }
    return {"status": "success", "tool": "scene_validator", "result": validation}
