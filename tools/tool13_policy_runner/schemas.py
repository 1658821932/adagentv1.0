"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    config_path: NonEmptyStr = Field(description='Existing AlgEngine model configuration path.')
    checkpoint_path: NonEmptyStr = Field(description='Existing driving-policy checkpoint path.')
    model_name: NonEmptyStr = Field(description='Experiment label for this policy.')
    data_type: NonEmptyStr = Field(description='Native WorldEngine scenario data type.')
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
        return self

class Output(OutputModel):
    result_dir: NonEmptyStr = Field(description='Closed-loop output directory containing official evaluation artifacts.')
    reports: list[Any] = Field(description='Native execution reports.')
    policy_result: NonEmptyStr = Field(description='Policy-run result identifier or path.')
    rollout_dir: NonEmptyStr | None = Field(default=None, description='WorldEngine openscene_format directory for rollout-based training.')
    reward_dir: NonEmptyStr | None = Field(default=None, description='Dense candidate-reward directory including actual plan_idx.csv when collected.')
    candidate_rewards_present: bool = Field(default=False, description='Whether candidate rewards were exported by this run.')
