"""Agent tool contract for scene_editor; backend execution lives in its adapter."""

from mcp_servers.dggt.adapter.service import edit_scene


def run(input_data: dict) -> dict:
    return {
        "status": "success",
        "tool": "scene_editor",
        "result": edit_scene(input_data),
    }
