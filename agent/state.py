"""
Agent runtime state.

AgentState stores the state of one Agent run. It does not make decisions or
execute tools; it only records task progress, execution history and lifecycle
status.
"""

from typing import Any, Dict, Optional


TOOL_STAGE_MAP = {
    "failure_analyzer": "failure_analysis",
    "scene_requirement_generator": "requirement_generation",
    "scene_conceiver": "scenario_conception",
    "video_retriever": "retrieval",
    "instance_discovery": "instance_discovery",
    "reconstruction": "reconstruction",
    "edit_plan_builder": "editing",
    "edit_config_generator": "editing",
    "scene_editor": "editing",
    "scene_validator": "validation",
    "recovery_decider": "recovery",
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

    def add_tool_result(self, tool_name: str, result: Dict[str, Any]) -> None:
        """Record a tool execution result."""

        self.history.append(
            {
                "tool": tool_name,
                "result": result,
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
        """Keep large per-frame detections in logs, not in every LLM prompt."""

        state = self.to_dict()
        history = []
        for entry in self.history:
            if entry.get("tool") != "instance_discovery":
                history.append(entry)
                continue
            tool_result = entry.get("result", {})
            result = tool_result.get("result", {})
            instances = []
            for instance in result.get("instances", []):
                frames = instance.get("frames", [])
                sample_indices = sorted({0, len(frames) // 2, len(frames) - 1}) if frames else []
                instances.append({
                    "instance_id": instance.get("instance_id"),
                    "frame_count": len(frames),
                    "sample_frames": [frames[index] for index in sample_indices],
                })
            history.append({
                **entry,
                "result": {
                    **tool_result,
                    "result": {**result, "instances": instances},
                },
            })
        return {**state, "history": history}
