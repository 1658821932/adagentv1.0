"""Agent tool contract for reconstruction; backend execution lives in its adapter."""

from mcp_servers.dggt.adapter.service import reconstruct_video


def run(input_data: dict) -> dict:
    return {
        "status": "success",
        "tool": "reconstruction",
        "result": reconstruct_video(input_data),
    }
