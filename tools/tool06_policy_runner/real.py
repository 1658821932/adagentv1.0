"""Agent tool contract for policy_runner; backend execution lives in its adapter."""

from mcp_servers.worldengine.adapter.service import run_policy, run_pdm
from tools.tool06_policy_runner.schemas import Input


def run(input_data: dict) -> dict:
    payload = Input.model_validate(input_data).model_dump(exclude_none=True, exclude={"attack_result"})
    if payload.get("execution_mode") == "pdm_closed_loop":
        try:
            result = run_pdm(payload)
        except Exception as error:
            return {"status": "error", "tool": "policy_runner", "error": {"type": "pdm_challenge_failed", "message": str(error)[:300]}}
        return {"status": "success", "tool": "policy_runner", "result": result}
    return {
        "status": "success",
        "tool": "policy_runner",
        "result": run_policy(payload),
    }
