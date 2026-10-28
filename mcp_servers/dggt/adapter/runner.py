"""JSON subprocess entry point for the DGGT reconstruction adapter."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from dotenv import load_dotenv

from .service import reconstruct


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        payload = json.load(sys.stdin)
        video_path = payload.get("video_path")
        if not isinstance(video_path, str) or not video_path.strip():
            raise ValueError("video_path must be a non-empty string")

        edit_config_path = payload.get("edit_config_path")
        if edit_config_path is not None and (not isinstance(edit_config_path, str) or not edit_config_path.strip()):
            raise ValueError("edit_config_path must be a non-empty string when provided")
        ego_trajectory_offsets_path = payload.get("ego_trajectory_offsets_path")
        if ego_trajectory_offsets_path is not None and (not isinstance(ego_trajectory_offsets_path, str) or not ego_trajectory_offsets_path.strip()):
            raise ValueError("ego_trajectory_offsets_path must be a non-empty string when provided")
        with redirect_stdout(sys.stderr):
            result = reconstruct(video_path.strip(), edit_config_path=edit_config_path, ego_trajectory_offsets_path=ego_trajectory_offsets_path, export_dynamic_frame_ply=payload.get("export_dynamic_frame_ply", False), num_frames=payload.get("num_frames"), export_render_state=payload.get("export_render_state", False))

        print(json.dumps({"status": "success", "result": result.to_dict()}, ensure_ascii=False))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
