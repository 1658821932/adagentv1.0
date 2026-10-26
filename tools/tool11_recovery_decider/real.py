"""LLM-backed recovery decision."""

from tools.llm_json import ask_json


def run(input_data: dict) -> dict:
    evaluation = input_data.get("evaluation") or input_data
    result = ask_json(input_data, "Return ONLY JSON with action (continue, retry, rollback, or replan), target_tool, reason, and modification. Choose rollback or replan when validation failed; continue only when passed is true.", evaluation)
    if result.get("action") not in {"continue", "retry", "rollback", "replan"}:
        raise ValueError("recovery_decider returned an unsupported action")
    return {"status": "success", "tool": "recovery_decider", "result": result}
