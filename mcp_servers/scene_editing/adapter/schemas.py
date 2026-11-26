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
    maneuver: Literal['stationary','moving'] | None = Field(default=None, description='Motion mode for an added vehicle (stationary or moving). Describe geometry/behavior in instruction.')
    instruction: Annotated[str, Field(min_length=1,max_length=1600)] = Field(description='Brief mechanism and qualitative relative placement/motion. Leave coordinates, speed, exact timing and control search to tool05; numeric evidence is advisory unless explicitly required by the user.')
    rationale: Annotated[str, Field(min_length=1,max_length=500)]
    uncertainty: Annotated[str, Field(min_length=1,max_length=500)]

    @model_validator(mode='after')
    def applicable_fields(self):
        if self.operation=='edit_existing' and self.maneuver is not None:
            raise ValueError('Existing-vehicle option does not use an added-vehicle maneuver')
        if self.operation=='add_vehicle' and self.target_track is not None:
            raise ValueError('Added-vehicle option cannot specify an existing target ID')
        return self


class EgoHistoryRoute(InputModel):
    end_xy_m: Annotated[list[float], Field(min_length=2, max_length=2)] = Field(description='Ego rear-axle [x,y] at 1.5s in scene-local metres. The recorded initial state is fixed.')
    end_yaw_rad: float | None = Field(default=None, description='Optional end heading in scene-local radians.')
    end_speed_mps: Annotated[float, Field(ge=0)] | None = Field(default=None, description='Optional continuous rear-axle speed at 1.5s.')
    via_xy_m: Annotated[list[float], Field(min_length=2, max_length=2)] | None = Field(default=None, description='Optional intermediate rear-axle point.')
    via_time_s: Annotated[float, Field(gt=0, lt=1.5)] | None = Field(default=None, description='Optional time for the intermediate point. Omit to allow any time inside the prefix.')
    connector_id: NonEmptyStr | None = Field(default=None, description='Optional scene-map connector used as route guidance.')

    @model_validator(mode='after')
    def timed_via_point(self):
        if self.via_time_s is not None and self.via_xy_m is None:
            raise ValueError('via_time_s requires via_xy_m')
        return self


class AttackInput(InputModel):
    scene_ref: SceneReference | NonEmptyStr = Field(description="Full source-scene reference or saved retrieval-result JSON path.")
    instruction: NonEmptyStr = Field(description="Qualitative vehicle-edit request for the selected source scene.")
    operation: Literal["auto", "edit_existing", "add_vehicle", "ego_only"] = Field(description="auto lets the SubAgent select/switch existing or added actors using plan_options. Explicit operations remain binding caller requirements.")
    plan_options: Annotated[list[AttackPlanOption],Field(max_length=3)] = Field(default_factory=list,description="Unranked coarse suggestions, not hard task contracts. The SubAgent chooses one or adapts it within the same scene/task/budget.")
    target_track: NonEmptyStr | None = Field(default=None, description="Optional existing-vehicle track ID from annotations. If absent, the backend must select an evidenced track from scene annotations before editing; never guess a role ID or frame number.")
    maneuver: Literal["stationary", "moving"] | None = Field(default=None, description="Required for add_vehicle: stationary or moving.")
    vehicle: VehicleAsset | None = None
    vehicle_catalog_class: Literal["any", "car", "bus", "truck"] | None = Field(
        default=None, description="Use native-size catalog assets for added vehicles. Select any when the class is unspecified.")
    prepare_ego_history: bool = Field(default=True,description='Stage A automatically prepares and freezes feasible forward acceleration during 0–1.5s before the attack Agent selects/edits a vehicle. No additional Agent/API call. Explicit false preserves the source history.')
    ego_history_target_speed_mps: Annotated[float, Field(ge=0)] | None = Field(default=None,description='Optional stage-A target speed at 1.5s. Omit for a bounded programmatic acceleration preference. Explicit targets are checked, never silently weakened.')
    ego_history_route: EgoHistoryRoute | None = Field(default=None, description='Optional short pose edit for stage A. Provide a reachable endpoint at 1.5s, with optional heading, speed, one intermediate point and connector guidance.')
    constraints: AttackConstraints = Field(default_factory=AttackConstraints)
    timeout: PositiveInt = Field(default=3600, description='Whole tool execution timeout, including selection/export. This is not the trajectory-stage budget.')
    stage_timeout_s: PositiveInt = Field(default=3600, description='Trajectory editing budget excluding time waiting for native VADv2 probes. Whole-tool timeout and shared probe/API/constructor budgets still apply; retry does not reset them.')
    task_contract: dict | None = Field(default=None, description="Optional user-supplied or explicitly approved attack_task.v2 hard contract with provenance. Do not convert the Agent's own coarse-plan preferences into this field.")
    acceptance_policy: dict | None = Field(default=None, description="Optional named research acceptance profile and explicit overrides.")
    revision_feedback: dict | None = Field(default=None, description="Compact review of the previous attack: stage, outcome, guidance and measured evidence including policy_behavior_summary (stopped_detected, progress_ratio_vs_recorded, signed_yaw_change_deg and task-specific failure_goal_met) and expert motion when relevant. Preserve the original instruction/contract; do not attach archived conversations or reasoning.")
    reference_kinematics: dict[str, Any] | None = Field(default=None, description="Optional structured kinematics from upstream failure analysis (ego/other speed, yaw change, stationary flag, size_m).")
    reference_bev_image: NonEmptyStr | None = Field(default=None, description="Optional upstream failure BEV image path used as visual imitation reference during object selection.")
    spatial_hint: dict[str, Any] | None = Field(default=None, description="Optional approximate attack_start_xy_m and conflict_zone_xy_m in initial-ego frame from Tool 04.")
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
        if not self.prepare_ego_history and self.ego_history_target_speed_mps is not None:
            raise ValueError('ego_history_target_speed_mps requires prepare_ego_history=true')
        if self.ego_history_route is not None and (not self.prepare_ego_history or self.ego_history_target_speed_mps is not None):
            raise ValueError('ego_history_route requires prepare_ego_history=true and uses its own optional end_speed_mps')
        if self.vehicle is not None and self.vehicle_catalog_class is not None:
            raise ValueError('Specify either an explicit vehicle or a catalog class')
        has_vehicle = self.vehicle is not None or self.vehicle_catalog_class is not None
        ids=[option.option_id for option in self.plan_options]
        if len(ids)!=len(set(ids)):raise ValueError('plan_options option_id must be unique')
        if self.operation=='auto':
            if self.target_track is not None or self.maneuver is not None:
                raise ValueError('auto cannot lock target_track or maneuver; express preferences in plan_options')
            if not self.plan_options:raise ValueError('auto requires 1..3 coarse plan_options')
            if not has_vehicle and all(option.operation=='add_vehicle' for option in self.plan_options):
                raise ValueError('auto with only added-vehicle options requires a supplied vehicle asset')
            return self
        if self.plan_options:raise ValueError('plan_options are for auto; explicit operations keep their original fixed semantics')
        if self.operation == "ego_only":
            if has_vehicle or self.maneuver is not None or self.target_track is not None:
                raise ValueError("ego_only rejects vehicle, maneuver and target_track; it runs Stage A ego history preparation and exports the scene directly")
        elif self.operation == "edit_existing":
            if has_vehicle or self.maneuver is not None:
                raise ValueError("edit_existing rejects vehicle/maneuver; select a real annotated track internally when target_track is omitted")
        elif not has_vehicle or not self.maneuver or self.target_track is not None:
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
    operation: Literal["edit_existing", "add_vehicle", "ego_only"] | None = Field(default=None, description="Attack operation used to create the frozen scene.")
    target_track: NonEmptyStr | None = None
    selected_candidate_id: NonEmptyStr
    plan_choice: dict[str, Any] | None = Field(default=None,description="Actual object/operation choice, associated suggestion and reason. Does not assert the suggested plan was followed exactly.")
    shared_history: dict[str, Any] | None = Field(default=None,description='Frozen common 0–1.5s history, speed request/evidence and manifest path. Expert and VAD branches must preserve it; target speeds are distinguished from delivered sampled takeover speed.')
    frozen_scene_path: NonEmptyStr
    frames_path: NonEmptyStr
    trajectory_path: NonEmptyStr | None = None
    provenance_path: NonEmptyStr
    submission_path: NonEmptyStr
    checks: StageChecks
    annotation_changes: dict[str, Any]
    simulation_input: SimulationInput | None = Field(default=None, description="Prepared exact edited native scene/asset paths for tool06; null if export-report explains unavailable assets/conversion. A frozen metadata directory alone is not executable.")
    native_export: dict[str, Any] | None = Field(default=None, description="Native export status, reason and report_path. Inspect this when simulation_input is null; never guess an executable folder.")
    rendered_videos: dict[str, NonEmptyStr] = Field(default_factory=dict, description="Camera name to edited render path; empty until rendered. Original recordings must not be relabelled as edited renders.")
    pdm_probe_result_path: NonEmptyStr | None = Field(default=None, description="Completed self-check bound to the submitted candidate; tool06 verifies it against the frozen scene before reuse.")

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
    target_track: NonEmptyStr | None = None
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
    expert_feedback: dict[str, Any] | None = Field(default=None,description='Measured independent expert clearance/motion headroom and an optional same-task revision_feedback for tool05. Advisory, no new acceptance gate. Preserve the original task, shared history and cumulative budgets; at most one margin revision.')
    ilft_training_data: ILTrainingData | None = Field(default=None, description="Converted expert labels for tool08 ilft. CSV alone is not ann_file; null until export is implemented.")
    render_status: dict[str, Any] | None = Field(default=None, description="Optional visualization status for this expert trajectory.")
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
