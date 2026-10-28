"""JSON stdin/stdout entry point for WorldEngine."""

from __future__ import annotations

import json
import sys

from .service import run_simulation


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        result = run_simulation(
            payload["config_path"], payload["checkpoint_path"], payload["model_name"],
            payload["data_type"], payload.get("react_type", "NR"), payload.get("asset_name"),
            payload.get("added_name", "mcp"), payload.get("num_scenarios"),
            payload.get("scenario_data_folder_path"), payload.get("asset_folder_path"),
            payload.get("scenario_pkl_file_name"),
            payload,
        )
        print(json.dumps({"status": "success", "result": result}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}, ensure_ascii=False))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
