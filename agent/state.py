"""
Agent runtime state.

AgentState stores the state of one Agent run. It does not make decisions or
execute tools; it only records task progress, execution history and lifecycle
status.
"""

from typing import Any, Dict, Optional
from copy import deepcopy


TOOL_STAGE_MAP = {
    "attack_vehicle_editor": "attack_editing",
    "ego_trajectory_editor": "ego_editing",
    "failure_analyzer": "failure_analysis",
    "scene_instance_generator": "requirement_generation",
    "video_retriever": "retrieval",
    "scene_validator": "validation",
    "policy_runner": "simulation",
    "failure_case_filter": "failure_filtering",
    "policy_trainer": "training",
}


class AgentState:
    """State container for a single Agent run."""

    def __init__(self, task: str):
        self.task = task
        self.history = []
        self.current_step = 0
        self.current_stage = "planning"
        self.status = "running"
        self.error: Optional[Any] = None

    def set_step(self, step: int) -> None:
        """Update the current Agent step."""

        self.current_step = step

    def set_stage_from_tool(self, tool_name: str) -> None:
        """Update the current execution stage from the selected tool."""

        self.current_stage = TOOL_STAGE_MAP.get(tool_name, "tool_execution")

    def add_decision(self, step: int, decision: Dict[str, Any]) -> None:
        """Record an LLM decision."""

        self.history.append(
            {
                "step": step,
                "decision": decision,
            }
        )

    def add_tool_result(self, tool_name: str, result: Dict[str, Any], result_ref: str | None = None) -> None:
        """Record a tool execution result."""

        self.history.append(
            {
                "tool": tool_name,
                "result": result,
                **({"result_ref": result_ref} if result_ref else {}),
            }
        )

    def mark_finished(self) -> None:
        """Mark the current Agent run as successfully finished."""

        self.status = "finished"
        self.current_stage = "completed"
        self.error = None

    def mark_error(self, error: Any) -> None:
        """Mark the current Agent run as failed."""

        self.status = "error"
        self.current_stage = "error"
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the current state."""

        return {
            "task": self.task,
            "history": self.history,
            "current_step": self.current_step,
            "current_stage": self.current_stage,
            "status": self.status,
            "error": self.error,
        }

    def to_orchestrator_dict(self) -> Dict[str, Any]:
        """Compact repeated editing context; leave complete state/log files intact."""
        state = deepcopy(self.to_dict())
        references = []
        for entry in state["history"]:
            decision = entry.get("decision", {})
            arguments = decision.get("arguments", {})
            name = decision.get("tool_name")
            if name == "attack_vehicle_editor":
                value = arguments.get("scene_ref")
                for tool, payload, reference in reversed(references):
                    if tool == "video_retriever" and isinstance(value, dict) and value == payload.get("scene_ref"):
                        arguments["scene_ref"] = reference
                        break
            elif name == "ego_trajectory_editor":
                value = arguments.get("attack_result")
                if isinstance(value, dict) and value.get("tool") == "attack_vehicle_editor" and value.get("status") == "success":
                    value = value.get("result")
                for tool, payload, reference in reversed(references):
                    if tool == "attack_vehicle_editor" and isinstance(value, dict) and value == payload:
                        arguments["attack_result"] = reference
                        break
            tool = entry.get("tool")
            envelope = entry.get("result", {})
            payload = envelope.get("result")
            if not tool or envelope.get("status") != "success" or not entry.get("result_ref") or not isinstance(payload, dict):
                continue
            references.append((tool, deepcopy(payload), entry["result_ref"]))
            if tool == "attack_vehicle_editor":
                payload.pop("source_scene_ref", None)
            elif tool == "failure_analyzer":
                # A duplicate of the native result; retain interpreted evidence/paths.
                payload.pop("raw", None)
        return state
