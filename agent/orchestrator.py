"""
LLM Orchestrator.

Responsible for asking the LLM to select the next tool according to the
current task, available tools and execution history.
"""

import json
from typing import Any, Dict, List, Tuple

from agent.llm_client import LLMClient
from agent.schemas import validate_decision
from agent.tool_descriptions import get_tool_descriptions


class Orchestrator:
    """Use an LLM to decide the next tool call."""

    def __init__(
        self,
        llm_client: LLMClient = None,
        decision_correction_attempts: int = 1,
    ):
        if (
            not isinstance(decision_correction_attempts, int)
            or isinstance(decision_correction_attempts, bool)
            or decision_correction_attempts < 0
        ):
            raise ValueError(
                "decision_correction_attempts must be a non-negative integer"
            )

        self.llm_client = llm_client or LLMClient()
        self.decision_correction_attempts = decision_correction_attempts
        self.tools = get_tool_descriptions()
        self.tool_names = {tool["name"] for tool in self.tools}

    def _build_prompt(self, task: str, state: Dict[str, Any]) -> list:
        system_prompt = """
You are an autonomous driving data generation agent.

Your responsibility is to select the most suitable next action according to
the current task and execution state.

You must output ONLY valid JSON.

Available actions:
1. call_tool
2. finish

For call_tool, output:
{
    "action": "call_tool",
    "tool_name": "tool name",
    "arguments": {},
    "reason": "why this tool is needed"
}

For finish, output:
{
    "action": "finish",
    "reason": "task completed"
}

Important:
- Carefully check the execution history before selecting the next action.
- Each tool's detailed usage rules are in its entry in available_tools
  (description / when_to_use / inputs / outputs). Read the relevant tool
  entry before calling it and follow its parameter constraints exactly.
- Use only facts, case IDs, instance IDs, file paths, and measurements present
  in the user's task or successful tool results. Never invent supporting case
  IDs, file paths, or claim a collision/validation/training result from a
  failure-reason summary alone. Leave unsupported evidence fields empty.
- Pass file paths and IDs between tools via the execution state exactly as
  returned by earlier tools; do not retype or guess them.
- Do not repeatedly call tools whose goals have already been achieved.
- If the original task goal has been completed and the required outputs have
  been generated, return finish immediately.
- For scenario generation tasks, once a valid edited scenario has been
  created and scene validation has passed, the task can be considered
  completed unless the user explicitly requests simulation or training.
"""

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task,
                        "available_tools": self.tools,
                        "current_state": state,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]

    def _parse_and_validate(
        self,
        response: str,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Parse an LLM response and validate the Agent decision."""

        try:
            decision = json.loads(response)
        except json.JSONDecodeError:
            return {}, ["LLM output is not valid JSON"]

        valid, errors = validate_decision(
            decision,
            self.tool_names,
        )
        if valid:
            return decision, []

        return decision, errors

    @staticmethod
    def _build_correction_message(errors: List[str]) -> Dict[str, str]:
        """Build one correction request for an invalid decision."""

        error_text = "\n".join(f"- {error}" for error in errors)
        return {
            "role": "user",
            "content": (
                "Your previous decision was invalid.\n\n"
                "Validation errors:\n"
                f"{error_text}\n\n"
                "Return ONLY one corrected JSON decision using the required "
                "call_tool or finish format."
            ),
        }

    def decide(self, task: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the LLM for the next valid action."""

        conversation = self._build_prompt(task, state)

        response = self.llm_client.chat(conversation)
        decision, errors = self._parse_and_validate(response)

        if not errors:
            return decision

        for _ in range(self.decision_correction_attempts):
            conversation.extend(
                [
                    {
                        "role": "assistant",
                        "content": response,
                    },
                    self._build_correction_message(errors),
                ]
            )

            response = self.llm_client.chat(conversation)
            decision, errors = self._parse_and_validate(response)

            if not errors:
                return decision

        return {
            "action": "error",
            "error_type": "decision_validation_failed",
            "validation_errors": errors,
            "raw_response": response,
            "reason": (
                "LLM decision failed validation after configured "
                "correction attempts"
            ),
        }
