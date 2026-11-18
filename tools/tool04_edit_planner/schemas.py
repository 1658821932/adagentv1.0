"""Tool04: road/space suitability review of a retrieved scene for one tool02 candidate."""
from typing import Any, Literal
from pydantic import Field, RootModel, model_validator
from common.artifacts import load_tool_result
from common.models import OutputModel
from common.scene_refs import SceneReference
from tools.llm_runtime import ToolModel, NonEmptyStr
from tools.tool02_scene_instance_generator.schemas import Attack, Output as ScenarioOutput

SCENARIO_TOOL = "scene_instance_generator"
RETRIEVAL_TOOL = "video_retriever"
CheckResult = Literal["pass", "fail", "unknown"]


class RetrievalContext(OutputModel):
    scene_ref: SceneReference
    candidate_id: NonEmptyStr = Field(description="tool02 candidate this retrieval was launched for; must match the selected candidate.")
    video_path: NonEmptyStr | None = None
    source_file: NonEmptyStr | None = None
    source_segment: dict[str, float] | None = None
    mode: str | None = None


def _unwrap(value, tool):
    """Accept a saved result_ref, a success envelope or a bare result object."""
    if isinstance(value, str):
        return load_tool_result(value, tool)
    if isinstance(value, dict):
        if "tool" in value or ("status" in value and "result" in value):
            if value.get("tool") != tool or value.get("status") != "success":
                raise ValueError("Expected a successful " + tool + " envelope")
            return value.get("result")
    return value


def normalize_retrieval(value):
    from pathlib import Path

    if isinstance(value, str):
        # A saved result_ref or success envelope is authoritative; only hand-typed
        # bare results are checked against the files that must exist on disk.
        return load_tool_result(value, RETRIEVAL_TOOL)
    if isinstance(value, dict) and ("tool" in value or ("status" in value and "result" in value)):
        if value.get("tool") != RETRIEVAL_TOOL or value.get("status") != "success":
            raise ValueError("Expected a successful " + RETRIEVAL_TOOL + " envelope")
        return value.get("result")
    if isinstance(value, dict):
        scene = value.get("scene_ref")
        if isinstance(scene, dict):
            missing = [k for k in ("frames_path", "cameras_path", "scenario_pkl")
                       if isinstance(scene.get(k), str) and not Path(scene[k]).exists()]
            if missing:
                raise ValueError(
                    "retrieval_result scene_ref paths do not exist on disk: "
                    + ", ".join(f"{k}={scene[k]}" for k in missing)
                    + ". Pass the video_retriever result or result_ref verbatim as retrieval_result.")
    return value


def normalize_scenario(value):
    return _unwrap(value, SCENARIO_TOOL)


class Input(ToolModel):
    scenario_result: ScenarioOutput = Field(
        description="Tool02 successful envelope, result object or result_ref, validated against tool02's current Output.")
    candidate_id: NonEmptyStr = Field(description="Candidate id selected from scenario_result.candidates.")
    retrieval_result: RetrievalContext = Field(
        description="Tool03 successful envelope, result object or result_ref for the same candidate.")

    @model_validator(mode="before")
    @classmethod
    def resolve_references(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            if "scenario_result" in value:
                value["scenario_result"] = normalize_scenario(value["scenario_result"])
            if "retrieval_result" in value:
                value["retrieval_result"] = normalize_retrieval(value["retrieval_result"])
        return value

    @model_validator(mode="after")
    def candidate_must_match(self):
        candidates = [item.candidate_id for item in self.scenario_result.candidates]
        if self.candidate_id not in candidates:
            raise ValueError(f"candidate_id {self.candidate_id!r} is not one of the scenario candidates")
        retrieved = self.retrieval_result.candidate_id
        if retrieved != self.candidate_id:
            raise ValueError(
                f"retrieval_result.candidate_id {retrieved!r} does not match the selected candidate {self.candidate_id!r}")
        return self

    @property
    def scenario(self):
        return self.scenario_result

    @property
    def candidate(self):
        return next(item for item in self.scenario_result.candidates if item.candidate_id == self.candidate_id)

    @property
    def scene_ref(self):
        return self.retrieval_result.scene_ref


class Checks(ToolModel):
    road: CheckResult = Field(description="Does the road layout support building the target scene? pass / fail / unknown.")
    lanes: CheckResult = Field(description="Do the lanes support it? pass / fail / unknown.")
    space: CheckResult = Field(description="Is there usable space for the planned construction? pass / fail / unknown.")


class AdaptedPlan(ToolModel):
    attack_edit_strategy: NonEmptyStr | None = Field(
        description="Adjusted other-vehicle edit for this real scene, or null when no other-vehicle edit is used.")
    ego_input_edit_strategy: NonEmptyStr | None = Field(
        description="Adjust ego approach speed and timing to meet the edited attack vehicle at the conflict while preserving meaningful speed and task difficulty; null only when the recorded input already produces that synchronized conflict.")
    ego_correct_trajectory_strategy: NonEmptyStr = Field(
        description="How the ego should drive to handle the danger safely in this scene.")


class Response(ToolModel):
    suitability: Literal["ready", "unsuitable", "needs_information"]
    reasons: list[NonEmptyStr] = Field(min_length=1, description="Why this verdict holds, from the observed scene.")
    checks: Checks = Field(description="Road, lanes and space verdicts; all three are required.")
    adapted_plan: AdaptedPlan | None = Field(description="Adjusted three-part plan, or null when the scene is not ready.")
    retrieval_suggestion: NonEmptyStr | None = Field(description="What to retrieve instead; required when unsuitable.")

    @model_validator(mode="after")
    def verdict_matches_checks(self):
        values = (self.checks.road, self.checks.lanes, self.checks.space)
        if self.suitability == "ready":
            if any(value != "pass" for value in values):
                raise ValueError("ready requires road, lanes and space to all pass")
            if self.adapted_plan is None:
                raise ValueError("ready requires an adapted plan")
        elif self.adapted_plan is not None:
            raise ValueError("non-ready results cannot carry an adapted plan")
        if self.suitability == "unsuitable":
            if "fail" not in values:
                raise ValueError("unsuitable requires at least one failing check")
            if not self.retrieval_suggestion:
                raise ValueError("unsuitable requires a retrieval suggestion")
        if self.suitability == "needs_information" and "fail" in values:
            raise ValueError("a confirmed structural failure is unsuitable, not needs_information")
        return self


class Output(Response):
    scene_id: NonEmptyStr
    scene_token: NonEmptyStr
    candidate_id: NonEmptyStr = Field(description="Selected tool02 candidate, copied by the tool.")
    target_scene: NonEmptyStr = Field(description="Copied from tool02; the tool never rewrites it.")
    attack: Attack = Field(description="Copied from tool02; the tool never rewrites it.")
    evidence: dict[str, Any] = Field(description="Observed scene facts used for the review.")

    @model_validator(mode="after")
    def bound_values(self):
        if self.attack.other_origin == "none" and self.adapted_plan is not None:
            if self.adapted_plan.attack_edit_strategy is not None:
                raise ValueError("no target vehicle in the target scene: attack_edit_strategy must be null")
        return self


class ViewRequest(ToolModel):
    action: Literal["request_views"]
    cameras: list[NonEmptyStr] = Field(min_length=1, max_length=2, description="Actual camera names from the source scene, e.g. CAM_L0 or CAM_R0.")
    purpose: NonEmptyStr = Field(description="What scene uncertainty needs these additional views.")


class Decision(RootModel[Response | ViewRequest]):
    """Internal visual request or final review; the public Output stays unchanged."""
