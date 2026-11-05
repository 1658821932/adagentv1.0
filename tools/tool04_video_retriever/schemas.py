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

class CameraReference(OutputModel):
    video_path: NonEmptyStr | None = None
    frame_count: int
    available_images: int
    images_complete: bool

class SceneReference(OutputModel):
    scene_id: NonEmptyStr
    scene_token: NonEmptyStr
    sample_root: NonEmptyStr
    dataset: NonEmptyStr
    matched_camera: NonEmptyStr
    frame_count: int
    fps: float | None = None
    manifest_path: NonEmptyStr
    frames_path: NonEmptyStr
    cameras_path: NonEmptyStr
    scenario_pkl: NonEmptyStr | None = None
    asset_config_path: NonEmptyStr | None = None
    asset_path: NonEmptyStr | None = None
    runtime_asset_path: NonEmptyStr | None = None
    maps_path: NonEmptyStr | None = None
    cameras: dict[str, CameraReference]
    images_complete: bool

class Output(OutputModel):
    video_path: NonEmptyStr
    match_score: float
    mode: NonEmptyStr
    source_file: NonEmptyStr | None = None
    source_segment: dict[str, float] | None = None
    scene_ref: SceneReference | None = None
