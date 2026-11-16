"""Public editing_stage.v1 contracts, based on the editing module report."""
from typing import Any, Annotated, Literal
from pydantic import Field, RootModel, model_validator
from common.models import InputModel, OutputModel, NonEmptyStr
from common.scene_refs import SceneReference
from common.artifacts import load_tool_result

PositiveInt = Annotated[int, Field(gt=0)]
CheckState = Literal["passed", "failed", "not_checked"]

class VehicleAsset(InputModel):
    asset_path: NonEmptyStr
    size_m: Annotated[list[Annotated[float, Field(gt=0)]], Field(min_length=3, max_length=3)] = Field(description="Length, width, height in metres; independently specified from the PLY asset.")

class AttackConstraints(InputModel):
    no_ego_contact_until_s: Annotated[float, Field(ge=0)] = Field(default=0.0,description='Additional no-contact delay. Zero adds no extra delay beyond the configured attack event window (default 1.5–7.5s); it does not change WE history or the earliest edit time.')
    target_window_s: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.target_window_s is not None and not (0 <= self.target_window_s[0] < self.target_window_s[1]):
            raise ValueError("target_window_s must be increasing non-negative seconds")
        return self

class AttackPlanOption(InputModel):
    option_id: NonEmptyStr
    operation: Literal['edit_existing','add_vehicle']
    target_track: NonEmptyStr | None = None
    maneuver: Literal['stationary','left_cut_in'] | None = None
    instruction: Annotated[str, Field(min_length=1,max_length=1600)]
    rationale: Annotated[str, Field(min_length=1,max_length=500)]
    uncertainty: Annotated[str, Field(min_length=1,max_length=500)]

    @model_validator(mode='after')
    def applicable_fields(self):
        if self.operation=='edit_existing' and self.maneuver is not None:
            raise ValueError('Existing-vehicle option does not use an added-vehicle maneuver')
        if self.operation=='add_vehicle' and (self.target_track is not None or self.maneuver is None):
            raise ValueError('Added-vehicle option needs maneuver and no existing target ID')
        return self


class AttackInput(InputModel):
    scene_ref: SceneReference | NonEmptyStr = Field(description="Full source-scene reference or saved retrieval-result JSON path.")
    instruction: NonEmptyStr = Field(description="Qualitative vehicle-edit request for the selected source scene.")
    operation: Literal["auto", "edit_existing", "add_vehicle"] = Field(description="auto lets the SubAgent select/switch existing or added actors using plan_options. Explicit operations remain binding caller requirements.")
    plan_options: Annotated[list[AttackPlanOption],Field(max_length=3)] = Field(default_factory=list,description="Unranked coarse suggestions, not hard task contracts. The SubAgent chooses one or adapts it within the same scene/task/budget.")
    target_track: NonEmptyStr | None = Field(default=None, description="Optional existing-vehicle track ID from annotations. If absent, the backend must select an evidenced track from scene annotations before editing; never guess a role ID or frame number.")
    maneuver: Literal["stationary", "left_cut_in"] | None = Field(default=None, description="Required for add_vehicle; the documented new-vehicle constructors have different internal signatures.")
    vehicle: VehicleAsset | None = None
    constraints: AttackConstraints = Field(default_factory=AttackConstraints)
    timeout: PositiveInt = Field(default=3600, description='Whole tool execution timeout, including selection/export. This is not the trajectory-stage budget.')
    stage_timeout_s: PositiveInt = Field(default=420, description='Trajectory editing wall-clock budget. Passing candidates retain a bounded final review/submission phase. Shared API and constructor budgets do not reset on retry.')
    vadv2_self_test: bool = Field(default=False, description='Enable the attack SubAgent native VADv2 collision loop. Off by default: the editor only guarantees geometric and task checks and returns a frozen scene; verify the attack afterwards with policy_runner(execution_mode=native). Does not call the legacy PDM planner.')
    vadv2_probe_budget: Annotated[int, Field(ge=1,le=5)] = Field(default=3, description='Cumulative native policy tests across candidate edits and object reselection. Geometric/constructor checks happen before these expensive tests.')
    task_contract: dict | None = Field(default=None, description="Optional user-supplied or explicitly approved attack_task.v2 hard contract with provenance. Do not convert the Agent's own coarse-plan preferences into this field.")
    acceptance_policy: dict | None = Field(default=None, description="Optional named research acceptance profile and explicit overrides.")
    revision_feedback: dict | None = Field(default=None, description="Compact review of the previous attack: stage, outcome, guidance and measured evidence including policy_behavior_summary (stopped_detected, progress_ratio_vs_recorded, signed_yaw_change_deg and task-specific failure_goal_met) and expert motion when relevant. Preserve the original instruction/contract; do not attach archived conversations or reasoning.")
    reuse_completed_from: NonEmptyStr | None = Field(default=None, description="Prior attack service run. Reuse verified completed object_selection and verified trajectory candidates only when scene, instruction, assets and constraints match; recompute cleanly if incomplete or stale.")

    @model_validator(mode="before")
    @classmethod
    def resolve_scene_reference(cls, value):
        if isinstance(value, dict) and isinstance(value.get("scene_ref"), str):
            value = dict(value)
            result = load_tool_result(value["scene_ref"], "video_retriever")
            if not isinstance(result.get("scene_ref"), dict):
                raise ValueError("Referenced video_retriever result has no scene_ref object")
            value["scene_ref"] = result["scene_ref"]
        return value

    @model_validator(mode="after")
    def operation_arguments(self):
        ids=[option.option_id for option in self.plan_options]
        if len(ids)!=len(set(ids)):raise ValueError('plan_options option_id must be unique')
        if self.operation=='auto':
            if self.target_track is not None or self.maneuver is not None:
                raise ValueError('auto cannot lock target_track or maneuver; express preferences in plan_options')
            if not self.plan_options:raise ValueError('auto requires 1..3 coarse plan_options')
            if not self.vehicle and all(option.operation=='add_vehicle' for option in self.plan_options):
                raise ValueError('auto with only added-vehicle options requires a supplied vehicle asset')
            return self
        if self.plan_options:raise ValueError('plan_options are for auto; explicit operations keep their original fixed semantics')
        if self.operation == "edit_existing":
            if self.vehicle is not None or self.maneuver is not None:
                raise ValueError("edit_existing rejects vehicle/maneuver; select a real annotated track internally when target_track is omitted")
        elif not self.vehicle or not self.maneuver or self.target_track is not None:
            raise ValueError("add_vehicle requires vehicle and maneuver; the backend assigns target_track")
        return self

class StageChecks(OutputModel):
    reference_screen: Literal["passed"]
    checks: dict[str, CheckState] = Field(min_length=1, description="Only actually evaluated checks may be passed; skipped checks are not_checked.")
    report_path: NonEmptyStr
    physical_avoidance: Literal["undetermined"] = "undetermined"

    @model_validator(mode="after")
    def consistent_acceptance(self):
        if "failed" in self.checks.values() or "passed" not in self.checks.values():
            raise ValueError("An accepted candidate requires at least one passed check and no failed checks")
        return self

class SimulationInput(InputModel):
    scenario_data_folder_path: NonEmptyStr
    asset_folder_path: NonEmptyStr
    scenario_pkl_file_name: NonEmptyStr = "all_scenarios.pkl"
    # The exported scene states its own horizon so a rollout never invents one.
    frame_count: PositiveInt | None = None

class ILTrainingData(InputModel):
    ann_file: NonEmptyStr
    token_file: NonEmptyStr

class AttackOutput(OutputModel):
    schema_version: Literal["editing_stage.v1"] = "editing_stage.v1"
    stage: Literal["attack_editing"] = "attack_editing"
    status: Literal["completed"] = "completed"
    mode: Literal["real", "mock"]
    run_dir: NonEmptyStr
    source_scene_ref: SceneReference
    scene_id: NonEmptyStr
    scene_token: NonEmptyStr
    operation: Literal["edit_existing", "add_vehicle"] | None = Field(default=None, description="Attack operation used to create the frozen scene.")
    target_track: NonEmptyStr
    selected_candidate_id: NonEmptyStr
    plan_choice: dict[str, Any] | None = Field(default=None,description="Actual object/operation choice, associated suggestion and reason. Does not assert the suggested plan was followed exactly.")
    frozen_scene_path: NonEmptyStr
    frames_path: NonEmptyStr
    trajectory_path: NonEmptyStr
    provenance_path: NonEmptyStr
    submission_path: NonEmptyStr
    checks: StageChecks
    annotation_changes: dict[str, Any]
    simulation_input: SimulationInput | None = Field(default=None, description="Prepared exact edited native scene/asset paths for tool07; null if export-report explains unavailable assets/conversion. A frozen metadata directory alone is not executable.")
    native_export: dict[str, Any] | None = Field(default=None, description="Native export status, reason and report_path. Inspect this when simulation_input is null; never guess an executable folder.")
    rendered_videos: dict[str, NonEmptyStr] = Field(default_factory=dict, description="Camera name to edited render path; empty until rendered. Original recordings must not be relabelled as edited renders.")
    pdm_probe_result_path: NonEmptyStr | None = Field(default=None, description="Completed self-check bound to the submitted candidate; tool07 verifies it against the frozen scene before reuse.")
    vadv2_probe_result_path: NonEmptyStr | None = Field(default=None, description='Exact submitted candidate native VADv2 self-test evidence; reuse rather than repeating evaluation.')
    policy_collision_report: dict[str, Any] | None = Field(default=None, description='Measured native contact with the selected attack track after VAD takeover, independently of PDM total or road compliance.')

class EgoInput(InputModel):
    attack_result: AttackOutput | NonEmptyStr = Field(description="Full attack result, successful envelope or saved attack-result JSON path.")
    target_effect: NonEmptyStr = Field(description="One correct ego behavior for the target scene.")
    allow_safe_stop: bool = True
    require_route_progress: bool | None = Field(default=None, description="Motion-goal hint for Agent review; null infers from target_effect, true means continue/proceed and false means a deliberate-stop goal. Does not impose any progress percentage, turn-angle threshold or automatic completion verdict. Temporary waiting can be part of a continued-driving task; allow_safe_stop controls explicit stop permission.")
    use_foci: bool = True
    max_edit_candidates: Annotated[int, Field(ge=0)] | None = Field(default=None, description="Optional new-edit resource budget. Zero permits selecting, inspecting and submitting existing candidates only; acceptance checks stay unchanged.")
    reuse_completed_from: NonEmptyStr | None = Field(default=None, description="Prior ego service run. Reuse completed understanding, constructor and initial FOCI stages only when content-bound checkpoints match task/scene/policy/code; recompute missing or stale stages and start a fresh editor request.")
    export_only: bool = Field(default=False, description="Recheck and export the already completed expert in reuse_completed_from for the identical attack and target_effect. Does not call an Agent, constructor, FOCI or renderer. Requires reuse_completed_from.")
    timeout: PositiveInt = 3600

    @model_validator(mode="before")
    @classmethod
    def unwrap_attack(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            attack = value.get("attack_result")
            if isinstance(attack, str):
                attack = load_tool_result(attack, "attack_vehicle_editor")
                value["attack_result"] = attack
            if isinstance(attack, dict) and "tool" in attack:
                if attack.get("tool") != "attack_vehicle_editor" or attack.get("status") != "success":
                    raise ValueError("attack_result must be a successful attack_vehicle_editor result")
                value["attack_result"] = attack["result"]
        return value

class EgoOutput(OutputModel):
    schema_version: Literal["editing_stage.v1"] = "editing_stage.v1"
    stage: Literal["ego_editing"] = "ego_editing"
    status: Literal["completed"] = "completed"
    mode: Literal["real", "mock"]
    run_dir: NonEmptyStr
    scene_id: NonEmptyStr
    scene_token: NonEmptyStr
    attack_run_dir: NonEmptyStr
    frozen_scene_path: NonEmptyStr
    target_track: NonEmptyStr
    selected_candidate_id: NonEmptyStr
    selected_source: Literal["constructor", "foci", "agent_edit", "repair"]
    expert_trajectory_path: NonEmptyStr
    trajectory_metadata_path: NonEmptyStr
    conflict_brief_path: NonEmptyStr
    submission_path: NonEmptyStr
    checks: StageChecks
    annotation_changes: dict[str, Any]
    simulation_input: SimulationInput | None = None
    native_export: dict[str, Any] | None = None
    expert_motion_summary: dict[str, Any] | None = Field(default=None, description="Measured kinematics and route/turn progress of the selected expert trajectory relative to the recorded trajectory.")
    ilft_training_data: ILTrainingData | None = Field(default=None, description="Converted expert labels for tool09 ilft. CSV alone is not ann_file; null until export is implemented.")
    rendered_videos: dict[str, NonEmptyStr] = Field(default_factory=dict)


class BoundedSearchOutcome(OutputModel):
    """Business no-solution result; never constructed from API or execution errors."""
    schema_version: Literal["editing_stage.v1"] = "editing_stage.v1"
    mode: Literal["real"] = "real"
    run_dir: NonEmptyStr
    reason: NonEmptyStr
    stop_reason: NonEmptyStr
    failed_requirements: list[str]
    failed_requirements_scope: Literal["observed_in_attempted_candidates"] = "observed_in_attempted_candidates"
    attempts: list[dict[str, Any]] = Field(min_length=1)
    evidence_paths: list[NonEmptyStr] = Field(min_length=1)
    scope: Literal["current_tools_and_budget"] = "current_tools_and_budget"
    global_infeasibility_proven: Literal[False] = False
    detail: dict[str, Any] = Field(default_factory=dict)

class AttackRefusal(BoundedSearchOutcome):
    stage: Literal["attack_editing"] = "attack_editing"
    status: Literal["unable_to_complete"] = "unable_to_complete"

class DefenseNotFound(BoundedSearchOutcome):
    stage: Literal["ego_editing"] = "ego_editing"
    status: Literal["defense_not_found"] = "defense_not_found"

class AttackResult(RootModel[Annotated[AttackOutput | AttackRefusal, Field(discriminator="status")]]):
    pass

class EgoResult(RootModel[Annotated[EgoOutput | DefenseNotFound, Field(discriminator="status")]]):
    pass
