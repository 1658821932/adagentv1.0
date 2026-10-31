"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    video_path: NonEmptyStr
    edit_config_path: NonEmptyStr | None = None
    ego_trajectory_offsets_path: NonEmptyStr | None = None
    camera_params_path: NonEmptyStr | None = None
    meters_to_dggt: PositiveFloat | None = None
    coordinate_system: Literal["smart_edit","dggt_world"] = "smart_edit"
    export_dynamic_frame_ply: bool = True
    num_frames: Literal[40] = 40
    timeout: PositiveInt = 86400

    @model_validator(mode="after")
    def editing_input(self):
        if not self.edit_config_path and not self.ego_trajectory_offsets_path:
            raise ValueError("An edit config or ego trajectory offsets file is required")
        return self

class Output(OutputModel):
    scene_ply: NonEmptyStr
    rendered_video: NonEmptyStr
    output_dir: NonEmptyStr
