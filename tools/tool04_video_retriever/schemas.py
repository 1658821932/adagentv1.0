"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    requirement: NonEmptyStr = Field(description='Natural-language source-video search description.')
    library: Literal['default', 'waymo'] = Field(default='default', description='Local index selection: default library or the separate Waymo library.')
    timeout: PositiveInt = Field(default=1800, description='Retrieval process timeout in seconds.')
    mock_video_path: NonEmptyStr | None = Field(default=None, description='Mock-only video path; it does not provide a real indexed scene.')

from common.scene_refs import CameraReference, SceneReference

class Output(OutputModel):
    video_path: NonEmptyStr = Field(description='Exported search clip path.')
    match_score: float = Field(description='Retrieval similarity score, not scenario-validation confidence.')
    mode: NonEmptyStr = Field(description='Retrieval execution mode.')
    source_file: NonEmptyStr | None = Field(default=None, description='Original indexed camera video.')
    source_segment: dict[str, float] | None = Field(default=None, description='Index-match start/end in source-video seconds; not exported clip bounds.')
    scene_ref: SceneReference | None = Field(default=None, description='Full source-scene identity, camera/image availability and annotation/Gaussian paths. Required by real retrieval; mock may return null. manifest_path locates ordered images, timestamps and calibrations.')
