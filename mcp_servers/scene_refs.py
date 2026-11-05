"""Shared dataset scene references used by retrieval and editing adapters."""
from tools.contracts import OutputModel, NonEmptyStr

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
