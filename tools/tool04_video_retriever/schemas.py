"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    requirement: NonEmptyStr
    library: Literal["default","waymo"] = "default"
    timeout: PositiveInt = 1800
    mock_video_path: NonEmptyStr | None = None

from mcp_servers.scene_refs import CameraReference, SceneReference

class Output(OutputModel):
    video_path: NonEmptyStr
    match_score: float
    mode: NonEmptyStr
    source_file: NonEmptyStr | None = None
    source_segment: dict[str, float] | None = None
    scene_ref: SceneReference | None = None
