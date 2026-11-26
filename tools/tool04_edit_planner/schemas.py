"""Tool 04 contracts: Dual-BEV imitation planning and validated Tool 05 handoff."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from pydantic import Field, model_validator

from common.artifacts import load_tool_result
from common.models import OutputModel
from common.scene_refs import SceneReference
from mcp_servers.scene_editing.adapter.schemas import AttackInput, VehicleAsset
from tools.llm_runtime import NonEmptyStr, ToolModel
from tools.tool02_scene_instance_generator.schemas import FailureContext, Output as ScenarioOutput

SCENARIO_TOOL = "scene_instance_generator"
RETRIEVAL_TOOL = "video_retriever"
CheckResult = Literal["pass", "fail", "unknown"]


class RetrievalAlternative(OutputModel):
    rank: int = Field(ge=2, le=3, description="Distinct rerank fallback position.")
    scene_ref: SceneReference
    video_path: NonEmptyStr | None = None
    source_file: NonEmptyStr | None = None
    source_segment: dict[str, float] | None = None
    match_score: float | None = None


class RetrievalContext(OutputModel):
    scene_ref: SceneReference
    candidate_id: NonEmptyStr = Field(description="tool02 candidate this retrieval was launched for.")
    video_path: NonEmptyStr | None = None
    source_file: NonEmptyStr | None = None
    source_segment: dict[str, float] | None = None
    mode: str | None = None
    alternatives: list[RetrievalAlternative] = Field(default_factory=list, max_length=2)


def _unwrap(value, tool):
    if isinstance(value, str):
        return load_tool_result(value, tool)
    if isinstance(value, dict) and ("tool" in value or ("status" in value and "result" in value)):
        if value.get("tool") != tool or value.get("status") != "success":
            raise ValueError("Expected a successful " + tool + " envelope")
        return value.get("result")
    return value


def normalize_retrieval(value):
    if isinstance(value, str):
        return load_tool_result(value, RETRIEVAL_TOOL)
    if isinstance(value, dict) and ("tool" in value or ("status" in value and "result" in value)):
        if value.get("tool") != RETRIEVAL_TOOL or value.get("status") != "success":
            raise ValueError("Expected a successful " + RETRIEVAL_TOOL + " envelope")
        return value.get("result")
    if isinstance(value, dict):
        scenes = [value.get("scene_ref")]
        scenes.extend(item.get("scene_ref") for item in value.get("alternatives") or [] if isinstance(item, dict))
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            missing = [k for k in ("frames_path", "cameras_path", "scenario_pkl")
                       if isinstance(scene.get(k), str) and not Path(scene[k]).exists()]
            if missing:
                raise ValueError(
                    "retrieval_result scene_ref paths do not exist on disk: "
                    + ", ".join(f"{k}={scene[k]}" for k in missing)
                )
    return value


class Input(ToolModel):
    scenario_result: ScenarioOutput = Field(
        description="Tool02 successful envelope, result object or result_ref."
    )
    candidate_id: NonEmptyStr = Field(description="Candidate id selected from scenario_result.candidates.")
    retrieval_result: RetrievalContext = Field(
        description="Tool03 successful envelope, result object or result_ref for the same candidate."
    )
    vehicle_catalog_class: Literal["car", "bus", "truck"] | None = Field(
        default=None, description="Set only when the user explicitly requires this added-vehicle class.")
    vehicle: VehicleAsset | None = Field(
        default=None,
        description="Optional explicit vehicle asset override; otherwise the editor selects native-size catalog assets.",
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_references(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            if "scenario_result" in value:
                value["scenario_result"] = _unwrap(value["scenario_result"], SCENARIO_TOOL)
            if "retrieval_result" in value:
                value["retrieval_result"] = normalize_retrieval(value["retrieval_result"])
        return value

    @model_validator(mode="after")
    def candidate_must_match(self):
        candidates = [item.candidate_id for item in self.scenario_result.candidates]
        if self.candidate_id not in candidates:
            raise ValueError(f"candidate_id {self.candidate_id!r} is not one of the scenario candidates")
        if self.retrieval_result.candidate_id != self.candidate_id:
            raise ValueError(
                f"retrieval_result.candidate_id {self.retrieval_result.candidate_id!r} != {self.candidate_id!r}"
            )
        ranks = [item.rank for item in self.retrieval_result.alternatives]
        if len(set(ranks)) != len(ranks) or ranks != sorted(ranks):
            raise ValueError("retrieval_result.alternatives must have unique ascending ranks")
        return self

    @property
    def candidate(self):
        return next(item for item in self.scenario_result.candidates if item.candidate_id == self.candidate_id)

    @property
    def scene_ref(self):
        return self.retrieval_result.scene_ref


class Checks(ToolModel):
    road: CheckResult
    lanes: CheckResult
    space: CheckResult


class AttackPlanAnchor(ToolModel):
    start_xy: list[float] = Field(min_length=2, max_length=2, description="E0-relative [x_east, y_north] start on Figure 2.")
    rationale: NonEmptyStr = Field(description="Why this start lane pairs with this conflict zone.")


ConflictRole = Literal["oncoming", "crossing", "adjacent_cut_in", "lead_vehicle", "stationary"]

class AdaptedPlan(ToolModel):
    conflict_role: ConflictRole | None = None
    attack_edit_strategy: NonEmptyStr | None = None
    attack_plans: list[AttackPlanAnchor] | None = Field(
        default=None,
        min_length=1,
        max_length=3,
        description="1..3 attack start plans (start_xy, rationale) read directly from Figure 2 E0-relative grid.",
    )
    ego_input_edit_strategy: NonEmptyStr | None = None


class Output(ToolModel):
    suitability: Literal["ready", "unsuitable", "needs_information"]
    reasons: list[NonEmptyStr] = Field(min_length=1)
    checks: Checks
    target_window_s: list[float] | None = Field(default=None, min_length=2, max_length=2)
    adapted_plan: AdaptedPlan | None = None
    retrieval_suggestion: NonEmptyStr | None = None
    scene_id: NonEmptyStr
    scene_token: NonEmptyStr
    candidate_id: NonEmptyStr
    target_scene: NonEmptyStr
    failure_context: FailureContext
    retrieved_bev_image: str | None = None
    attack_input: dict[str, Any] | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def verdict_and_handoff_are_consistent(self):
        values = (self.checks.road, self.checks.lanes, self.checks.space)
        single_vehicle = self.failure_context.failure_driver in ("departure_ego_only", "collision_non_vehicle")
        if self.suitability == "ready":
            if any(v != "pass" for v in values):
                raise ValueError("ready requires road, lanes and space to all pass")
            if self.adapted_plan is None:
                raise ValueError("ready requires adapted_plan")
            if single_vehicle:
                if self.adapted_plan.attack_edit_strategy is not None or self.attack_input is not None:
                    raise ValueError("single-vehicle ready plan must have null attack_edit_strategy and null attack_input")
            else:
                if self.attack_input is None:
                    raise ValueError("ready vehicle-driven plan requires validated attack_input for tool05")
                handoff = AttackInput.model_validate(self.attack_input)
                ref = handoff.scene_ref if isinstance(handoff.scene_ref, SceneReference) else SceneReference.model_validate(handoff.scene_ref)
                if (ref.scene_id, ref.scene_token) != (self.scene_id, self.scene_token):
                    raise ValueError("attack_input.scene_ref differs from reviewed scene")
        else:
            if self.adapted_plan is not None or self.attack_input is not None or self.target_window_s is not None:
                raise ValueError("non-ready results cannot carry adapted_plan, target_window_s, or attack_input")
            if self.suitability == "unsuitable":
                if "fail" not in values or not self.retrieval_suggestion:
                    raise ValueError("unsuitable requires at least one fail check and retrieval_suggestion")
            if self.suitability == "needs_information" and "fail" in values:
                raise ValueError("confirmed fail check must be unsuitable, not needs_information")
        return self
