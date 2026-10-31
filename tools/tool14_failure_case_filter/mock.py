"""Local mock implementation of failure-case filtering."""


def run(input_data: dict) -> dict:
    """Return mock filtering output without writing to a database."""
    return {
        "status": "success",
        "tool": "failure_case_filter",
        "result": {
            "is_failure_case": False,
            "collision": False,
            "failure_cases": [],
        },
    }
