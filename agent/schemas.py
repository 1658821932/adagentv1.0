"""
Validation helpers for Agent decisions and Tool results.

These validators check only shared outer structures. They intentionally do not
validate tool-specific argument fields or business result fields.
"""

from typing import Any, Iterable, List, Tuple


VALID_ACTIONS = {"call_tool", "finish", "stop"}
VALID_TOOL_STATUSES = {"success", "error"}


def validate_decision(
    decision: Any,
    available_tools: Iterable[str],
) -> Tuple[bool, List[str]]:
    """Validate the outer structure of an Agent decision."""

    errors: List[str] = []

    if not isinstance(decision, dict):
        return False, ["decision must be a JSON object"]

    action = decision.get("action")
    if action not in VALID_ACTIONS:
        errors.append("action must be one of: call_tool, finish, stop")

    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reason must be a non-empty string")

    if action == "call_tool":
        tool_name = decision.get("tool_name")
        arguments = decision.get("arguments")

        if not isinstance(tool_name, str) or not tool_name.strip():
            errors.append("tool_name must be a non-empty string")
        elif tool_name not in set(available_tools):
            errors.append(f"unknown tool_name: {tool_name}")

        if not isinstance(arguments, dict):
            errors.append("arguments must be a JSON object")

    return len(errors) == 0, errors


def validate_tool_result(
    tool_result: Any,
    expected_tool_name: str,
) -> Tuple[bool, List[str]]:
    """Validate the shared outer structure returned by a Tool."""

    errors: List[str] = []

    if not isinstance(tool_result, dict):
        return False, ["tool result must be a JSON object"]

    status = tool_result.get("status")
    if status not in VALID_TOOL_STATUSES:
        errors.append("status must be one of: success, error")

    tool_name = tool_result.get("tool")
    if not isinstance(tool_name, str) or not tool_name.strip():
        errors.append("tool must be a non-empty string")
    elif tool_name != expected_tool_name:
        errors.append(
            f"tool mismatch: expected {expected_tool_name}, got {tool_name}"
        )

    if status == "success" and "result" not in tool_result:
        errors.append("success result must contain result")

    if status == "error":
        error = tool_result.get("error")
        if not isinstance(error, dict):
            errors.append("error result must contain an error object")
        else:
            error_type = error.get("type")
            message = error.get("message")
            if not isinstance(error_type, str) or not error_type.strip():
                errors.append("error.type must be a non-empty string")
            if not isinstance(message, str) or not message.strip():
                errors.append("error.message must be a non-empty string")

    return len(errors) == 0, errors
