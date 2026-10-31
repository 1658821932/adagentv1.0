"""Filter decision failures using custom-scene geometric evaluation.

This evaluator covers dynamic-object collisions. Gaussian/static reconstruction
geometry and road-boundary compliance are not evaluated.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
from mcp_servers.process import run_logged
import tempfile
from pathlib import Path
from typing import Any


from mcp_servers.runtime import runtime_task, current_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]
from .config import SIMENGINE_PYTHON
from .schemas import EvaluationResult


EVALUATOR = r'''
import json, pickle, math, sys

scenario_path = sys.argv[1]
metadata_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "" else None
payload = pickle.load(open(scenario_path, "rb"))
scene = next(iter(payload.values())) if isinstance(payload, dict) else payload[0]
tracks = scene.get("object_track", {})
ego = tracks.get("ego", {}).get("state", {})
ego_pos = ego.get("position", [])
ego_len = ego.get("length", [])
ego_wid = ego.get("width", [])
valid = ego.get("valid", [True] * len(ego_pos))
trajectory_source = "scenario_object_track"
if metadata_path:
    metadata = pickle.load(open(metadata_path, "rb"))
    if isinstance(metadata, list):
        actual = []
        rear_axle_to_center = float(scene.get("metadata", {}).get("rear_axle_to_center_dist", 0.0))
        for frame in metadata:
            if not isinstance(frame, dict) or frame.get("ego2global_translation") is None:
                continue
            translation = frame["ego2global_translation"]
            pose = frame["ego2global"]
            center = [float(translation[axis] + rear_axle_to_center * pose[axis][0]) for axis in range(3)]
            actual.append((int(frame["frame_idx"]), center))
        if actual:
            actual.sort()
            if [index for index, _ in actual] != list(range(len(actual))):
                raise ValueError("WorldEngine metadata frames must be contiguous from zero")
            ego_pos = [position for _, position in actual]
            ego = dict(ego)
            ego["position"] = ego_pos
            valid = [True] * len(ego_pos)
            trajectory_source = "worldengine_openscene_metadata"

def scalar(values, i, default):
    try:
        value = values[i]
        if hasattr(value, "item"): value = value.item()
        return float(value)
    except (IndexError, TypeError, ValueError):
        return float(default)

def xy(values, i):
    value = values[i]
    return float(value[0]), float(value[1])

def collision(a, b, i):
    ax, ay = xy(a["position"], i)
    bx, by = xy(b["position"], i)
    return (abs(ax - bx) <= (scalar(a.get("length", []), i, 4.5) + scalar(b.get("length", []), i, 4.5)) / 2
            and abs(ay - by) <= (scalar(a.get("width", []), i, 2.0) + scalar(b.get("width", []), i, 2.0)) / 2)

dynamic_hits = []
dynamic_tracks = [(name, value.get("state", {})) for name, value in tracks.items() if name != "ego" and isinstance(value, dict) and value.get("state")]

for i, pos in enumerate(ego_pos):
    if not scalar(valid, i, 1):
        continue
    for name, other in dynamic_tracks:
        if i < len(other.get("position", [])) and scalar(other.get("valid", []), i, 1) and collision(ego, other, i):
            dynamic_hits.append({"frame": i, "object_id": name})
print(json.dumps({
    "collision": bool(dynamic_hits),
    "collision_events": dynamic_hits,
    "frames_evaluated": len(ego_pos),
    "trajectory_source": trajectory_source,
}, ensure_ascii=False))
'''


@runtime_task("worldengine", "evaluation")
def filter_failure_cases(input_data: dict[str, Any]) -> EvaluationResult:
    scenario_path = input_data.get("scenario_pkl") or input_data.get("scenario_path")
    if not isinstance(scenario_path, str) or not scenario_path.strip():
        raise ValueError("failure_case_filter requires scenario_pkl")
    scenario = Path(scenario_path).expanduser().resolve()
    if not scenario.is_file():
        raise FileNotFoundError(f"scenario_pkl does not exist: {scenario}")
    python = Path(input_data.get("python") or os.environ.get("SIMENGINE_PYTHON") or SIMENGINE_PYTHON)
    if not python.is_file():
        raise FileNotFoundError(f"SimEngine Python does not exist: {python}")
    result_dir = input_data.get("result_dir")
    metadata_file = None
    if isinstance(result_dir, str) and result_dir.strip():
        candidates = sorted((Path(result_dir).expanduser().resolve() / "WE_output/openscene_format/meta_datas").glob("*.pkl"))
        if not candidates:
            raise FileNotFoundError("WorldEngine result has no executed ego trajectory metadata")
        metadata_file = candidates[0]
    try:
        process = run_logged(
            [str(python), "-c", EVALUATOR, str(scenario), str(metadata_file or "")],
            cwd=PROJECT_ROOT, log_path=current_run("worldengine").logs / "evaluation.log", json_stdout=True, timeout=int(input_data.get("timeout",120)),
        )
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode,process.args,process.stdout,process.stderr)
        evaluation = json.loads(process.stdout)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "decision evaluation failed") from exc

    reasons = []
    if evaluation["collision"]:
        reasons.append("collision")
    failure = {
        "scenario_pkl": str(scenario),
        "reasons": reasons,
        "evaluation": evaluation,
    }
    return {
            "run_dir": str(current_run("worldengine").root),
            "is_failure_case": bool(reasons),
            "failure_cases": [failure] if reasons else [],
            **evaluation,
        }
