"""Public editing_stage.v1 contracts, based on the editing module report."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator
from tools.contracts import InputModel, OutputModel, NonEmptyStr
from mcp_servers.scene_refs import SceneReference

PositiveInt = Annotated[int, Field(gt=0)]
CheckState = Literal["passed", "failed", "not_checked"]

class VehicleAsset(InputModel):
    asset_path: NonEmptyStr
    size_m: Annotated[list[Annotated[float, Field(gt=0)]], Field(min_length=3, max_length=3)] = Field(description="Length, width, height in metres; independently specified from the PLY asset.")

class AttackConstraints(InputModel):
    no_ego_contact_until_s: Annotated[float, Field(ge=0)] = 3.0
    target_window_s: Annotated[list[float], Field(min_length=2, max_length=2)] | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.target_window_s is not None and not (0 <= self.target_window_s[0] < self.target_window_s[1]):
            raise ValueError("target_window_s must be increasing non-negative seconds")
        return self

class AttackInput(InputModel):
    scene_ref: SceneReference = Field(description="Pass tool04.result.scene_ref, retaining its dataset identity, metadata and asset paths.")
    instruction: NonEmptyStr = Field(description="Selected tool02 candidate.edit_strategy or an explicit user edit request.")
    operation: Literal["edit_existing", "add_vehicle"]
    target_track: NonEmptyStr | None = Field(default=None, description="Optional existing-vehicle track ID from annotations. If absent, the backend must select an evidenced track from scene annotations before editing; never guess a role ID or frame number.")
    maneuver: Literal["stationary", "left_cut_in"] | None = Field(default=None, description="Required for add_vehicle; the documented new-vehicle constructors have different internal signatures.")
    vehicle: VehicleAsset | None = None
    constraints: AttackConstraints = Field(default_factory=AttackConstraints)
    timeout: PositiveInt = 3600

    @model_validator(mode="after")
    def operation_arguments(self):
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
    target_track: NonEmptyStr
    selected_candidate_id: NonEmptyStr
    frozen_scene_path: NonEmptyStr
    frames_path: NonEmptyStr
    trajectory_path: NonEmptyStr
    provenance_path: NonEmptyStr
    submission_path: NonEmptyStr
    checks: StageChecks
    annotation_changes: dict[str, Any]
    simulation_input: SimulationInput | None = Field(default=None, description="Prepared native scene/asset paths for tool13; null until converted. A frozen metadata directory alone is not executable.")
    rendered_videos: dict[str, NonEmptyStr] = Field(default_factory=dict, description="Camera name to edited render path; empty until rendered. Original recordings must not be relabelled as edited renders.")

class EgoInput(InputModel):
    attack_result: AttackOutput = Field(description="Pass the preceding attack_vehicle_editor.result or its successful full envelope. Frozen scene and target identity are retained.")
    target_effect: NonEmptyStr = Field(description="Pass tool02.result.target_effect, the one global correct ego behaviour.")
    allow_safe_stop: bool = True
    use_foci: bool = True
    timeout: PositiveInt = 3600

    @model_validator(mode="before")
    @classmethod
    def unwrap_attack(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            attack = value.get("attack_result")
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
    ilft_training_data: ILTrainingData | None = Field(default=None, description="Converted expert labels for tool15 ilft. CSV alone is not ann_file; null until export is implemented.")
    rendered_videos: dict[str, NonEmptyStr] = Field(default_factory=dict)
