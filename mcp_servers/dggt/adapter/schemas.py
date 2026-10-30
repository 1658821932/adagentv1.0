"""Public reconstruction and editing artifact contract."""
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class ReconstructionResult:
    video_path: str
    run_dir: str
    input_dir: str
    input_scene_name: str
    output_dir: str
    scene_ply: str
    static_ply: str
    dynamic_ply: str | None
    sky_ply: str
    camera_params: str
    render_camera_params: str
    supersplat_camera_poses: str
    rendered_video: str
    render_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
