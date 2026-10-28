"""Agent tool contract for failure_case_filter; backend execution lives in its adapter."""

from mcp_servers.worldengine.adapter.evaluation import filter_failure_cases


def run(input_data: dict) -> dict:
    return {
        "status": "success",
        "tool": "failure_case_filter",
        "result": filter_failure_cases(input_data),
    }
