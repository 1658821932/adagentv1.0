"""Agent tool contract for instance_discovery; backend execution lives in its adapter."""

from mcp_servers.dggt.adapter.service import discover_instances


def run(input_data: dict) -> dict:
    return {
        "status": "success",
        "tool": "instance_discovery",
        "result": discover_instances(input_data),
    }
