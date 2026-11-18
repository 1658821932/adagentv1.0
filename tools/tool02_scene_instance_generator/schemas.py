"""Detailed new scenario requirements inspired by one saved failure category."""
from typing import Any, Literal, Optional
from pydantic import Field, field_validator, model_validator
from common.artifacts import load_tool_result
from tools.llm_runtime import NonEmptyStr, ToolModel

# Same semantics as the tool01 attack block, declared locally and strictly.
MapContext = Literal["straight_road", "curve", "t_junction", "cross_intersection", "merge"]
EgoAction = Literal["straight", "turn_left", "turn_right"]
EgoLaneKeeping = Literal["left_offset", "keep", "right_offset", "unknown"]
OtherOrigin = Literal["ahead_same_lane", "behind_same_lane", "oncoming", "left_adjacent",
                      "right_adjacent", "left_arm", "right_arm", "none"]
OtherAction = Literal["straight", "turn_left", "turn_right", "none"]
OtherLaneKeeping = Literal["left_offset", "keep", "right_offset", "unknown", "none"]


SOURCE_FIELDS = ("token", "simulation", "map_location", "evidence_dir")


def normalized(text):
    return " ".join(text.casefold().split())


class Attack(ToolModel):
    """Two-vehicle structure to reproduce, with tool01's exact value set."""
    map_context: MapContext = Field(description="Road layout: straight_road / curve / t_junction / cross_intersection / merge.")
    ego_action: EgoAction = Field(description="Ego direction: straight / turn_left / turn_right.")
    ego_lane_keeping: EgoLaneKeeping = Field(description="Ego lateral position in its own lane: left_offset / keep / right_offset / unknown.")
    other_origin: OtherOrigin = Field(description="Where the other vehicle comes from, or none when no target vehicle exists.")
    other_action: OtherAction = Field(description="Other vehicle direction: straight / turn_left / turn_right, or none.")
    other_lane_keeping: OtherLaneKeeping = Field(description="Other vehicle lateral position in its own lane: left_offset / keep / right_offset / unknown, or none.")

    @model_validator(mode="after")
    def none_is_paired(self):
        if (self.other_origin == "none") != (self.other_action == "none"):
            raise ValueError("other_origin and other_action must both be none, or both describe a real vehicle")
        if (self.other_origin == "none") != (self.other_lane_keeping == "none"):
            raise ValueError("other_origin and other_lane_keeping must both be none, or both describe a real vehicle")
        return self


class Input(ToolModel):
    failure_analysis: dict[str, Any] | NonEmptyStr = Field(description="Successful failure_analyzer result, envelope or absolute result_ref. Only failure, attack, environment and verification reach the model; source fields remain in the run record for tracing.")

    @field_validator("failure_analysis", mode="before")
    @classmethod
    def normalize_analysis(cls, value):
        if isinstance(value, str):
            value = load_tool_result(value, "failure_analyzer")
        elif isinstance(value, dict) and ("tool" in value or ("status" in value and "result" in value)):
            if value.get("tool") not in (None, "failure_analyzer") or value.get("status") != "success":
                raise ValueError("failure_analysis envelope must be a successful failure_analyzer result")
            value = value.get("result")
        if not isinstance(value, dict):
            raise ValueError("failure_analysis must be a successful failure_analyzer result")
        failure = value.get("failure")
        if not isinstance(failure, dict) or not isinstance(failure.get("ego_cause"), str) or not failure["ego_cause"].strip():
            raise ValueError("failure_analysis requires the measured failure.ego_cause sentence")
        verification = value.get("verification")
        if not isinstance(verification, dict):
            raise ValueError("failure_analysis requires the final visual verification record")
        violation_step = value.get("violation_step")
        if isinstance(violation_step, bool) or not isinstance(violation_step, int):
            raise ValueError("failure_analysis requires the measured violation_step integer")
        try:
            attack = Attack.model_validate(value.get("attack"))
        except (TypeError, ValueError):
            raise ValueError("failure_analysis requires the final tool01 attack with road, action and lane-keeping fields") from None
        environment = value.get("environment")
        if not isinstance(environment, dict) or not isinstance(environment.get("time"), str) or not environment["time"].strip():
            raise ValueError("failure_analysis requires the measured environment.time")
        if "weather" in environment and environment["weather"] is not None and not isinstance(environment["weather"], str):
            raise ValueError("environment.weather must be text or null")
        source = {}
        # Second pass over an already normalized result (registry hands tools a dumped dict).
        if isinstance(value.get("source"), dict):
            for name in SOURCE_FIELDS:
                carried = value["source"].get(name)
                if carried is None:
                    continue
                if not isinstance(carried, str):
                    raise ValueError(f"failure_analyzer {name} must be text when provided")
                if carried.strip():
                    source[name] = carried
        for name in SOURCE_FIELDS:
            if name not in value or value[name] is None:
                continue
            if not isinstance(value[name], str):
                raise ValueError(f"failure_analyzer {name} must be text when provided")
            if value[name].strip():
                source[name] = value[name]
        normalized_analysis = {"violation_step": violation_step,
                               "failure": dict(failure), "attack": attack.model_dump(mode="json"),
                               "environment": dict(environment), "verification": dict(verification)}
        if source:
            normalized_analysis["source"] = source
        return normalized_analysis

class Candidate(ToolModel):
    candidate_id: NonEmptyStr = Field(description="Stable short id for this variant, e.g. cand_1. Unique within the response.")
    base_video_requirement: NonEmptyStr = Field(description="What the base video must contain once the later edits are subtracted, in one sentence.")
    attack_edit_strategy: Optional[NonEmptyStr] = Field(description="How the other vehicle is edited, or null when no other-vehicle edit is needed. The field is always present.")
    ego_input_edit_strategy: Optional[NonEmptyStr] = Field(description="How ego approach speed and timing are adjusted so ego and the edited attack vehicle meet at the conflict, preserving meaningful speed and difficulty; null only when the recorded input already produces that synchronized conflict. The field is always present.")
    ego_correct_trajectory_strategy: NonEmptyStr = Field(description="How the ego should drive afterwards to avoid the failure.")


class Response(ToolModel):
    target_scene: NonEmptyStr = Field(description="The whole target scene: environment, road layout, ego and other-vehicle motion and interaction, without subtracting the later edits.")
    attack: Attack = Field(description="Two-vehicle structure the scene reproduces.")
    candidates: list[Candidate] = Field(min_length=3, max_length=3, description="Exactly three variants of the same target scene.")

    @model_validator(mode="after")
    def candidate_ids_are_unique(self):
        ids = [item.candidate_id for item in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_id must be unique within one response")
        if self.attack.other_origin == "none" and any(
                item.attack_edit_strategy is not None for item in self.candidates):
            raise ValueError("no target vehicle in the upstream analysis: attack_edit_strategy must be null")
        return self


class Output(Response):
    pass
