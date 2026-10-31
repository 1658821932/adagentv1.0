"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    video_path: NonEmptyStr
    num_frames: Literal[40] = 40
    export_render_state: bool = False
    timeout: PositiveInt = 86400

class Output(OutputModel):
    scene_ply: NonEmptyStr
    input_dir: NonEmptyStr
    input_scene_name: NonEmptyStr
    output_dir: NonEmptyStr
    rendered_video: NonEmptyStr
