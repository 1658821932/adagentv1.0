"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    config_path: NonEmptyStr
    checkpoint_path: NonEmptyStr
    model_name: NonEmptyStr
    data_type: NonEmptyStr
    react_type: Literal["R","NR"] = "NR"
    asset_name: NonEmptyStr | None = None
    added_name: NonEmptyStr = "mcp"
    num_scenarios: PositiveInt | None = None
    scenario_data_folder_path: NonEmptyStr | None = None
    asset_folder_path: NonEmptyStr | None = None
    scenario_pkl_file_name: NonEmptyStr | None = None
    dggt_output_dir: NonEmptyStr | None = None
    instance_id: Annotated[int,Field(ge=0)] = 0
    translation_scale: PositiveFloat = 10000.0
    num_history: PositiveInt = 4
    num_future: PositiveInt | None = None
    max_frame_per_scene: Annotated[int,Field(ge=5)] = 40
    sampling_poses: PositiveInt = 8
    with_dense_reward_manager: bool = False
    ego_policy: NonEmptyStr | None = None
    ego_controller: NonEmptyStr | None = None
    ego_client: NonEmptyStr | None = None
    ego_navigation: NonEmptyStr | None = None
    agent_policy: NonEmptyStr | None = None
    agent_navigation: NonEmptyStr | None = None
    agent_client: NonEmptyStr | None = None
    timeout: PositiveInt = 86400

    @model_validator(mode="after")
    def horizon(self):
        if self.num_history >= self.max_frame_per_scene:
            raise ValueError("max_frame_per_scene must exceed num_history")
        if self.num_future is not None and self.num_history+self.num_future != self.max_frame_per_scene:
            raise ValueError("num_history + num_future must equal max_frame_per_scene")
        return self

class Output(OutputModel):
    result_dir: NonEmptyStr
    reports: list[Any]
    policy_result: NonEmptyStr
    rollout_dir: NonEmptyStr | None = None
    reward_dir: NonEmptyStr | None = None
    candidate_rewards_present: bool = False
