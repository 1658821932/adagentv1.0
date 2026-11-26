"""Agent tool contract for policy_trainer; backend execution lives in its adapter."""

from mcp_servers.worldengine.adapter.service import train_policy


def run(input_data: dict) -> dict:
    return {
        "status": "success",
        "tool": "policy_trainer",
        "result": train_policy(input_data),
    }
