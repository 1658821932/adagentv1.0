"""Agent tool contract for video_retriever; backend execution lives in its adapter."""

from mcp_servers.sentry_video_search.adapter.client import retrieve_video


def run(input_data: dict) -> dict:
    return {
        "status": "success",
        "tool": "video_retriever",
        "result": retrieve_video(input_data),
    }
