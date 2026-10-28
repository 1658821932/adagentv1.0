"""Agent tool contract for edit_config_generator; backend execution lives in its adapter."""

from mcp_servers.smart_edit.adapter.service import generate_edit_config


def run(input_data: dict) -> dict:
    return {
        "status": "success",
        "tool": "edit_config_generator",
        "result": generate_edit_config(input_data),
    }
