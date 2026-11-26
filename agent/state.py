"""
Agent runtime state.

AgentState stores the state of one Agent run. It does not make decisions or
execute tools; it only records task progress, execution history and lifecycle
status.
"""

from typing import Any, Dict, Optional


TOOL_STAGE_MAP = {
    "attack_vehicle_editor": "attack_editing",
    "failure_analyzer": "failure_analysis",
    "scene_instance_generator": "requirement_generation",
    "video_retriever": "retrieval",
    "edit_planner": "edit_planning",
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
        self.pending_job = None
        self.background_jobs = {}
        self.error: Optional[Any] = None
        self.current_request = task
        self.user_messages = []
        self.resume_context = None
        self.execution_scope = None

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
                "step": self.current_step,
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
            "pending_job": self.pending_job,
            "background_jobs": self.background_jobs,
            "current_request": self.current_request,
            "user_messages": self.user_messages,
            "resume_context": self.resume_context,
            "execution_scope": self.execution_scope,
        }

    def to_orchestrator_dict(self) -> Dict[str, Any]:
        """All-tool decision summaries; original state and result files stay intact."""
        from agent.context import compact_state
        return compact_state(self.to_dict())
