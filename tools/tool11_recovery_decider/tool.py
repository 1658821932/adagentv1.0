"""Local mock implementation of evaluation handling and recovery decisions."""


def run(input_data: dict) -> dict:
    """Select continuation or rollback from a mock validation result."""
    evaluation = input_data.get("evaluation", {})
    evaluation_result = evaluation.get("result", evaluation)
    passed = evaluation_result.get("passed", False)
    if passed:
        decision = {
            "action": "continue",
            "target_tool": 12,
            "reason": "scene validation passed",
            "modification": {},
        }
    else:
        decision = {
            "action": "rollback",
            "target_tool": 7,
            "reason": "scene validation failed",
            "modification": {"suggestion": "adjust edit plan"},
        }
    return {"status": "success", "tool": "recovery_decider", "result": decision}
