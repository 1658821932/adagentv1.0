"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    edit_plan: NonEmptyStr | dict[str, Any]
    output_path: NonEmptyStr
    output_kind: Literal["rigid_config","ego_trajectory_offsets"] = "rigid_config"
    instance_info_path: NonEmptyStr | None = None
    frame_count: Literal[40] = 40
    scene_info: dict[str, Any] | str | None = None
    asset_path: NonEmptyStr | None = None
    asset_scale: PositiveFloat | None = None
    timeout: PositiveInt = 1800

class Output(OutputModel):
    saved_path: NonEmptyStr
    tool_calls: list[str]
