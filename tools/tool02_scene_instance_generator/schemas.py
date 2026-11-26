"""Tool 02 contracts: target scene, 3 prioritized candidates with per-candidate asset requirements."""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import Field, field_validator, model_validator

from common.artifacts import load_tool_result
from tools.llm_runtime import NonEmptyStr, ToolModel

MapContext = Literal["straight_road", "curve", "t_junction", "cross_intersection", "merge", "roundabout"]
Outcome = Literal["collision", "road_departure"]
FailureDriver = Literal["collision_with_vehicle", "collision_non_vehicle", "departure_by_other", "departure_ego_only"]
VehicleClass = Literal["car", "bus", "truck"]

SOURCE_FIELDS = ("token", "simulation", "map_location", "evidence_dir", "bev_image")


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


class FailureContext(ToolModel):
    """Authoritative upstream failure classification passed through unchanged."""
    outcome: Outcome
    failure_driver: FailureDriver
    map_context: MapContext


class AssetRequirementInput(ToolModel):
    """LLM-declared asset class for an add-vehicle candidate (null if no added asset needed)."""
    vehicle_class: VehicleClass = Field(description="Required added-vehicle class: car / bus / truck.")


class AssetRequirement(ToolModel):
    """Final per-candidate added-vehicle asset requirement with measured size_m from Tool 01."""
    vehicle_class: VehicleClass
    size_m: list[float] = Field(default_factory=lambda: [4.432, 2.033, 1.637], min_length=3, max_length=3, description="Optional vehicle size_m; defaults to canonical native asset size.")


class Input(ToolModel):
    failure_analysis: dict[str, Any] | NonEmptyStr = Field(
        description="Successful failure_analyzer result, envelope or absolute result_ref."
    )

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

        violation_step = value.get("violation_step")
        if isinstance(violation_step, bool) or not isinstance(violation_step, int):
            raise ValueError("failure_analysis requires integer violation_step")

        failure = value.get("failure")
        if not isinstance(failure, dict):
            raise ValueError("failure_analysis requires failure object")
        FailureContext.model_validate({
            "outcome": failure.get("outcome"),
            "failure_driver": failure.get("failure_driver"),
            "map_context": failure.get("map_context"),
        })
        for key in ("ego_behavior", "summary"):
            if not isinstance(failure.get(key), str) or not failure[key].strip():
                raise ValueError(f"failure_analysis requires non-empty failure.{key}")
        if not isinstance(failure.get("other_behavior"), str):
            raise ValueError("failure_analysis requires string failure.other_behavior")

        kinematics = value.get("kinematics")
        if not isinstance(kinematics, dict) or not isinstance(kinematics.get("ego"), dict):
            raise ValueError("failure_analysis requires kinematics with ego measurements")
        if failure["failure_driver"] in ("collision_with_vehicle", "departure_by_other"):
            other_k = kinematics.get("other")
            if not isinstance(other_k, dict) or not isinstance(other_k.get("size_m"), list) or len(other_k["size_m"]) != 3:
                raise ValueError("vehicle-driven failure requires kinematics.other.size_m")

        environment = value.get("environment")
        if not isinstance(environment, dict) or not isinstance(environment.get("time"), str) or not environment["time"].strip():
            raise ValueError("failure_analysis requires environment.time")

        source = {}
        if isinstance(value.get("source"), dict):
            for name in SOURCE_FIELDS:
                v = value["source"].get(name)
                if isinstance(v, str) and v.strip():
                    source[name] = v
        for name in SOURCE_FIELDS:
            v = value.get(name)
            if isinstance(v, str) and v.strip():
                source[name] = v

        normalized_analysis = {
            "violation_step": violation_step,
            "failure": dict(failure),
            "kinematics": dict(kinematics),
            "environment": dict(environment),
            "verification": dict(value.get("verification") or {}),
        }
        if source:
            normalized_analysis["source"] = source
        return normalized_analysis


class CandidateResponse(ToolModel):
    candidate_id: NonEmptyStr = Field(description="Stable short id: cand_1, cand_2, cand_3.")
    base_video_requirement: NonEmptyStr = Field(
        description="What the base video must contain once the later edits are subtracted, in one sentence."
    )
    retrieval_query: NonEmptyStr = Field(
        min_length=6, max_length=48,
        description="Compact local-index query (6-48 chars): only weather/time when material, road form, ego direction and required clear space. Omit editing/conflict/other-vehicle-action terms.",
    )
    asset_requirement: Optional[AssetRequirementInput] = Field(
        description="Required added-vehicle asset class when adding a vehicle; null when editing an existing vehicle or single-vehicle scene."
    )
    attack_edit_strategy: Optional[NonEmptyStr] = Field(
        description="How the unique target vehicle is added/edited, or null for single-vehicle failures."
    )
    ego_input_edit_strategy: Optional[NonEmptyStr] = Field(
        description="How ego approach speed/timing is adjusted to synchronize the conflict, or null when recorded input suffices."
    )

    @field_validator("retrieval_query")
    @classmethod
    def retrieval_query_is_not_an_edit_description(cls, value: str) -> str:
        banned = (
            "速度", "时机", "同步", "同时", "交汇", "冲突", "碰撞", "危险",
            "编辑", "调整", "可改", "轨迹", "左偏", "右偏", "偏移",
            "保持车道", "他车", "其他车辆", "一辆车",
        )
        found = [term for term in banned if term in value]
        if found:
            raise ValueError("retrieval_query must omit downstream editing terms: " + ", ".join(found))
        return value

    @model_validator(mode="after")
    def asset_requires_attack(self):
        if self.asset_requirement is not None and self.attack_edit_strategy is None:
            raise ValueError("asset_requirement cannot be set when attack_edit_strategy is null")
        return self


class Candidate(CandidateResponse):
    asset_requirement: Optional[AssetRequirement] = Field(
        description="Required added-vehicle asset (vehicle_class + measured size_m) when adding a vehicle; null otherwise."
    )


class Response(ToolModel):
    target_scene: NonEmptyStr = Field(
        description="One sentence describing the full target scene (environment, minimal road layout, ego and unique target vehicle behavior)."
    )
    candidates: list[CandidateResponse] = Field(
        min_length=3, max_length=3,
        description="Exactly three distinct implementation variants, prioritizing add_vehicle first.",
    )

    @model_validator(mode="after")
    def candidate_ids_are_unique(self):
        ids = [item.candidate_id for item in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_id must be unique within one response")
        return self


class Output(ToolModel):
    target_scene: NonEmptyStr
    failure_context: FailureContext
    kinematics: dict[str, Any]
    source_bev_image: str | None = Field(default=None, description="Original Tool 01 failure BEV image path passed through for Tool 04 dual-BEV planning.")
    candidates: list[Candidate] = Field(min_length=3, max_length=3)
