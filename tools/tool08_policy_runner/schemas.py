"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import ConfigDict, Field, model_validator
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    model_config = ConfigDict(json_schema_extra={'required': []})

    execution_mode: Literal['native', 'pdm_closed_loop'] = Field(default='native', description='Native full simulation or the independent PDM closed-loop challenge.')
    config_path: NonEmptyStr | None = Field(default=None, description='Existing AlgEngine model configuration path.')
    checkpoint_path: NonEmptyStr | None = Field(default=None, description='Existing driving-policy checkpoint path.')
    model_name: NonEmptyStr | None = Field(default=None, description='Experiment label for this policy.')
    data_type: NonEmptyStr | None = Field(default=None, description='Native WorldEngine scenario data type.')
    challenge_run_baseline_diagnostic: bool = Field(default=False, description='Optionally simulate original scene for diagnosis only; no original-scene safety gate.')
    challenge_baseline_scene_path: NonEmptyStr | None = Field(default=None, description='Read-only original scene for timestamps, map and source integrity. It is not simulated by default.')
    challenge_attack_scene_path: NonEmptyStr | None = Field(default=None, description='Frozen attack scene used for the PDM challenge.')
    challenge_target_track: NonEmptyStr | None = Field(default=None, description='Attack target track present in the frozen scene.')
    challenge_target_is_new: bool = Field(default=False, description='Whether the attack introduced a new target track.')
    challenge_ignore_traffic_lights: bool = Field(default=False, description='Explicit signal-agnostic PDM experiment; original light annotations remain unchanged, signal compliance is not evaluated.')
    challenge_history_s: PositiveFloat = Field(default=3.0, description='Recorded history before PDM takeover.')
    challenge_probe_result_path: NonEmptyStr | None = Field(default=None, description='Optional accepted attack pdm_probe_result_path; reuse only after exact scene/history/target/runtime/map verification.')
    react_type: Literal['R', 'NR'] = Field(default='NR', description='Reactive R or nonreactive NR background traffic.')
    asset_name: NonEmptyStr | None = Field(default=None, description='Optional native renderer asset group.')
    added_name: NonEmptyStr = Field(default='mcp', description='Run-name suffix.')
    num_scenarios: PositiveInt | None = Field(default=None, description='Optional maximum number of scenarios to execute.')
    scenario_data_folder_path: NonEmptyStr | None = Field(default=None, description='Prepared native scenario directory; may come from editor simulation_input.')
    asset_folder_path: NonEmptyStr | None = Field(default=None, description='Prepared native Gaussian/renderer asset directory.')
    scenario_pkl_file_name: NonEmptyStr | None = Field(default=None, description='Scenario pickle name within scenario_data_folder_path.')
    num_history: PositiveInt = Field(default=4, description='Input history frame count.')
    num_future: PositiveInt | None = Field(default=None, description='Future frame count; derived from the total budget when omitted.')
    max_frame_per_scene: Annotated[int, Field(ge=5)] = Field(default=40, description='Total frame budget; num_history + num_future must equal this value.')
    sampling_poses: PositiveInt = Field(default=8, description='Native rollout timing override in 0.5-second units.')
    with_dense_reward_manager: bool = Field(default=False, description='Collect native 8192-candidate PDM rewards for subsequent RLFT.')
    ego_policy: NonEmptyStr | None = Field(default=None, description='Optional native ego policy override.')
    ego_controller: NonEmptyStr | None = Field(default=None, description='Optional native ego controller override.')
    ego_client: NonEmptyStr | None = Field(default=None, description='Optional native planner-client override.')
    ego_navigation: NonEmptyStr | None = Field(default=None, description='Optional native ego navigation override.')
    agent_policy: NonEmptyStr | None = Field(default=None, description='Optional native background-agent policy override.')
    agent_navigation: NonEmptyStr | None = Field(default=None, description='Optional native background-agent navigation override.')
    agent_client: NonEmptyStr | None = Field(default=None, description='Optional native background-agent client override.')
    timeout: PositiveInt = Field(default=86400, description='Simulation process timeout in seconds.')

    @model_validator(mode="after")
    def horizon(self):
        if self.num_history >= self.max_frame_per_scene:
            raise ValueError("max_frame_per_scene must exceed num_history")
        if self.num_future is not None and self.num_history+self.num_future != self.max_frame_per_scene:
            raise ValueError("num_history + num_future must equal max_frame_per_scene")
        if self.execution_mode == 'pdm_closed_loop':
            if not self.challenge_baseline_scene_path or not self.challenge_attack_scene_path or not self.challenge_target_track:
                raise ValueError('pdm_closed_loop requires baseline/attack scene paths and a target track')
        elif not self.config_path or not self.checkpoint_path or not self.model_name or not self.data_type:
            raise ValueError('Native execution requires config/checkpoint/model/data_type')
        return self

class Output(OutputModel):
    result_dir: NonEmptyStr = Field(description='Closed-loop output directory containing official evaluation artifacts.')
    reports: list[Any] = Field(description='Native execution reports.')
    policy_result: NonEmptyStr = Field(description='Policy-run result identifier or path.')
    rollout_dir: NonEmptyStr | None = Field(default=None, description='WorldEngine openscene_format directory for rollout-based training.')
    reward_dir: NonEmptyStr | None = Field(default=None, description='Dense candidate-reward directory including actual plan_idx.csv when collected.')
    candidate_rewards_present: bool = Field(default=False, description='Whether candidate rewards were exported by this run.')
    pdm_challenge: dict | None = Field(default=None, description='Independent baseline/attack PDM outcomes; not an official native PDM score.')
