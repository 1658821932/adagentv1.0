"""Agent tool contract for video_retriever; backend execution lives in its adapter."""

from mcp_servers.sentry_video_search.adapter.service import retrieve_video


def run(input_data: dict) -> dict:
    result = retrieve_video(input_data)
    # Echo the originating tool02 candidate so tool04 can verify that the plan and
    # the retrieved scene describe the same variant.
    if isinstance(result, dict) and input_data.get("candidate_id"):
        result = dict(result, candidate_id=input_data["candidate_id"])
    return {
        "status": "success",
        "tool": "video_retriever",
        "result": result,
    }
