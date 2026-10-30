"""Agent tool contract for policy_runner; backend execution lives in its adapter."""

from mcp_servers.worldengine.adapter.service import run_policy


def run(input_data: dict) -> dict:
    return {
        "status": "success",
        "tool": "policy_runner",
        "result": run_policy(input_data),
    }
