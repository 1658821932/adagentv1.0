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
- Use only facts, case IDs, instance IDs, file paths, and measurements present
  in the user's task or successful tool results. Never invent supporting case
  IDs or claim a collision/validation/training result from a failure-reason
  summary alone. Leave unsupported evidence fields empty. Proposed scenario
  parameters are design targets, not observed facts.
- For instance_discovery, pass reconstruction.result.input_dir as image_dir
  and reconstruction.result.input_scene_name as scene_name. The output scene
  directory (often 001) is not the preprocessed input scene (often 000).
- Agent DGGT reconstruction, instance discovery, edited rendering, and
  WorldEngine closed-loop simulation all use the first 40 frames. Pass
  num_frames=40, sequence_length=40, frame_count=40, and
  max_frame_per_scene=40 respectively; never request 64 frames.
- The instance_discovery state contains only sampled bbox frames. Pass the
  instance_info_path to Smart Edit for complete per-frame data; do not copy
  every detection frame into tool arguments.
- A detected instance may be moved only with a trajectory offset relative to
  its original per-frame motion. Do not describe it as a new object or use
  `add_dynamic_object` unless a real asset_path and explicit asset_scale are
  available in state. A natural-language behavior such as "trigger ego
  braking" does not change the ego vehicle: create a second
  edit_config_generator call with output_kind=ego_trajectory_offsets and then
  pass both the instance edit config and ego_trajectory_offsets_path to
  scene_editor.
- For policy_runner, use only paths that exist in tool results or the known
  WorldEngine installation. The standard project paths are
  `WORLDENGINE_ROOT/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py`
  and
  `WORLDENGINE_ROOT/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth`.
  Never invent a config under the Agent repository.
- Do not repeatedly call tools whose goals have already been achieved.
- If the original task goal has been completed and the required outputs have
  been generated, return finish immediately.
- For scenario generation tasks, once a valid edited scenario has been
  created and scene validation has passed, the task can be considered
  completed unless the user explicitly requests simulation or training.
- For DGGT instance trajectory edits, Smart Edit waypoints are translation
  offsets added to the instance's original position at each frame, not
  absolute positions. Account for the original per-frame motion: to reach a
  desired displacement relative to frame 0, subtract the original displacement
  from frame 0 at each frame. A constant offset after a cut-in does not make
  an originally moving vehicle stay in the target lane.
- DGGT instance-edit conversion has an empirical starting scale of about
  meters_to_dggt=0.01; 0.001 produces smaller shifts. Calibrate against the
  current reconstruction and rendered frames before treating either as valid.
  Never use WorldEngine's DGGT-to-metres translation_scale (often 10000) as
  meters_to_dggt for scene_editor; these serve different conversions. Do not
  treat a desired absolute distance ahead as an offset from the existing car.
- For a front-car cut-in edit, use the verified scene-specific target as a
  planning criterion: enter the ego lane by about frame 12, then hold the
  vehicle's ego-lane lateral position through frame 39 without continuing
  rightward. Do not merely repeat the frame-12 translation offset: compensate
  the source car's original per-frame motion. Keep the DGGT world-Y vehicle
  translation exactly zero; do not sink or lift the car. Check the rendered
  car against lane markings at early, middle, and late frames. A projected
  center near the camera principal point can guide this scene but is not
  sufficient to claim lane membership or geometry validity. DGGT edit-config
  translations use pre-normalization world units; when deriving corrections
  from normalized PLY coordinates, multiply by ply_world_transform.scale.
- YOLO instance_info provides 2D boxes, not a calibrated 3D vehicle track or
  road height. Do not claim exact metres-to-ego placement or road contact from
  it. Starting in the existing vehicle's lane usually means a zero initial
  lateral offset. Run visual validation on at least 8 uniformly spaced frames
  and treat bad grounding, off-road placement, or fragmented vehicle geometry
  as edit failure even if artifact files exist.
- DGGT ego_trajectory_offsets alter rendering camera poses only. They cannot
  force the WorldEngine closed-loop policy to brake. Judge actual braking from
  WorldEngine policy output, not from the edited video or the edit plan text.
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
