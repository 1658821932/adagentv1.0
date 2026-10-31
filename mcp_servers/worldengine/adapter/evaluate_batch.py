"""Match executed trajectories to scene identities before geometric filtering.

Executed metadata is mandatory when a result directory is supplied. The
collision calculation remains the existing axis-aligned approximation.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
import numbers
from pathlib import Path
import pickle
import re


def aliases(value):
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()
    if text.endswith(".pkl"):
        text = text[:-4]
    values = {text}
    suffix = re.search(r"(?:^|-)([0-9a-fA-F]{16})$", text)
    if suffix:
        values.add(suffix.group(1).lower())
    return values


def scenes_from(payload):
    if isinstance(payload, dict) and "object_track" in payload:
        rows = [(payload.get("id") or payload.get("scene_token") or payload.get("token") or "scene_0", payload)]
    elif isinstance(payload, dict):
        rows = list(payload.items())
    elif isinstance(payload, list):
        rows = [(scene.get("id") or scene.get("scene_token") or scene.get("token") or f"scene_{i}", scene)
                for i, scene in enumerate(payload) if isinstance(scene, dict)]
        if len(rows) != len(payload):
            raise ValueError("Each scenario must be a dictionary")
    else:
        raise ValueError("Expected a scene dictionary or scene list")
    if not rows:
        raise ValueError("Scenario inventory is empty")
    scenes, index = {}, defaultdict(set)
    for key, scene in rows:
        if not isinstance(scene, dict) or not isinstance(scene.get("object_track"), dict):
            raise ValueError(f"Invalid scenario: {key}")
        identity = str(scene.get("id") or key)
        if identity in scenes:
            raise ValueError(f"Duplicate scenario identity: {identity}")
        scenes[identity] = scene
        values = [key, identity, scene.get("scene_id"), scene.get("scenario_id"), scene.get("scene_token")]
        # Some list-based archives provide only token, with no id.
        if not scene.get("id"):
            values.append(scene.get("token"))
        for value in values:
            for alias in aliases(value):
                index[alias].add(identity)
    return scenes, index


def resolve(values, index, label):
    matches = []
    for value in values:
        found = set()
        for alias in aliases(value):
            found.update(index.get(alias, set()))
        if found:
            matches.append(found)
    if not matches:
        raise ValueError(f"Cannot match {label} to a source scenario")
    candidates = set.intersection(*matches)
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous or conflicting scene identities in {label}")
    return next(iter(candidates))


def metadata_by_scene(directory, index):
    paths = sorted(directory.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No executed ego trajectory metadata in {directory}")
    matched, sources = defaultdict(list), defaultdict(list)
    for path in paths:
        with path.open("rb") as stream:
            frames = pickle.load(stream)
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"Metadata must be a non-empty frame list: {path.name}")
        groups = defaultdict(list)
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError(f"Invalid metadata frame in {path.name}")
            token, log = frame.get("scene_token"), frame.get("log_name")
            groups[(str(token or ""), str(log or ""))].append(frame)
        # A combined file may contain many scenes; its filename is not an identity.
        include_filename = len(groups) == 1 and path.stem not in {"combined", "all_scenes"}
        for (token, log), group in groups.items():
            values = [token, log]
            if include_filename:
                values.append(path.stem)
            identity = resolve(values, index, path.name)
            matched[identity].extend(group)
            sources[identity].append(str(path))
    return matched, sources


def reported_scenes(result_dir, index):
    expected = set()
    for path in sorted(result_dir.rglob("runner_report_*.json")):
        rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            raise ValueError(f"Invalid runner report: {path}")
        for row in rows:
            if not isinstance(row, dict) or not row.get("scenario_name"):
                raise ValueError(f"Runner report has no scenario identity: {path}")
            expected.add(resolve([row["scenario_name"]], index, path.name))
    return expected


def scalar(values, i, default):
    try:
        value = values[i]
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (IndexError, TypeError, ValueError):
        return float(default)


def collision(a, b, i):
    ax, ay = map(float, a["position"][i][:2])
    bx, by = map(float, b["position"][i][:2])
    return (abs(ax-bx) <= (scalar(a.get("length", []), i, 4.5)+scalar(b.get("length", []), i, 4.5))/2
            and abs(ay-by) <= (scalar(a.get("width", []), i, 2.0)+scalar(b.get("width", []), i, 2.0))/2)


def evaluate_scene(identity, scene, frames=None):
    tracks = scene["object_track"]
    ego = dict(tracks.get("ego", {}).get("state", {}))
    positions = ego.get("position", [])
    valid = ego.get("valid", [True]*len(positions))
    trajectory_source = "scenario_object_track"
    if frames is not None:
        ordered = []
        rear_to_center = float(scene.get("metadata", {}).get("rear_axle_to_center_dist", 0.0))
        for frame in frames:
            i = frame.get("frame_idx")
            if isinstance(i, bool) or not isinstance(i, numbers.Integral):
                raise ValueError(f"{identity}: frame_idx must be an integer")
            translation, pose = frame.get("ego2global_translation"), frame.get("ego2global")
            if translation is None or pose is None:
                raise ValueError(f"{identity}: executed frame lacks ego pose")
            center = [float(translation[axis] + rear_to_center*pose[axis][0]) for axis in range(3)]
            if not all(math.isfinite(value) for value in center):
                raise ValueError(f"{identity}: ego positions must be finite")
            ordered.append((int(i), center))
        ordered.sort()
        if not ordered or [i for i, _ in ordered] != list(range(len(ordered))):
            raise ValueError(f"{identity}: metadata frames must be unique and contiguous from zero")
        positions = [p for _, p in ordered]
        ego["position"] = positions
        valid = [True]*len(positions)
        trajectory_source = "worldengine_openscene_metadata"
    if len(positions) == 0:
        raise ValueError(f"{identity}: ego trajectory is empty")
    dynamic = [(str(name), value["state"]) for name, value in tracks.items()
               if name != "ego" and isinstance(value, dict) and value.get("state")]
    hits = []
    for i, _ in enumerate(positions):
        if not scalar(valid, i, 1):
            continue
        for name, other in dynamic:
            if i < len(other.get("position", [])) and scalar(other.get("valid", []), i, 1) and collision(ego, other, i):
                hits.append({"scene_id": identity, "frame": i, "object_id": name})
    return {"scene_id": identity, "collision": bool(hits), "collision_events": hits,
            "frames_evaluated": len(positions), "trajectory_source": trajectory_source}


def evaluate_batch(scenario_path, result_dir=None, scene_token=None):
    with Path(scenario_path).open("rb") as stream:
        scenes, index = scenes_from(pickle.load(stream))
    selected = {resolve([scene_token], index, "requested scene_token")} if scene_token else None
    matched, sources = {}, {}
    if result_dir:
        result_dir = Path(result_dir)
        matched, sources = metadata_by_scene(result_dir/"WE_output/openscene_format/meta_datas", index)
        expected = reported_scenes(result_dir, index)
        if selected is None:
            selected = expected or set(scenes)
        missing = selected-set(matched)
        if missing:
            raise ValueError("Missing executed metadata for scenes: " + ", ".join(sorted(missing)))
    if selected is None:
        selected = set(scenes)
    results, failures, events = [], [], []
    for identity in sorted(selected):
        evaluation = evaluate_scene(identity, scenes[identity], matched[identity] if result_dir else None)
        evaluation["metadata_files"] = sorted(set(sources.get(identity, [])))
        results.append(evaluation)
        events.extend(evaluation["collision_events"])
        if evaluation["collision"]:
            failures.append({"scene_id": identity, "scenario_pkl": str(scenario_path),
                             "reasons": ["collision"], "evaluation": evaluation})
    return {"is_failure_case": bool(failures), "failure_cases": failures,
            "scene_count": len(results), "scene_results": results,
            "evaluated_scene_ids": [row["scene_id"] for row in results],
            "skipped_scene_ids": sorted(set(scenes)-selected),
            "collision": bool(events), "collision_events": events,
            "frames_evaluated": sum(row["frames_evaluated"] for row in results),
            "trajectory_source": "worldengine_openscene_metadata" if result_dir else "scenario_object_track"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_path")
    parser.add_argument("--result-dir")
    parser.add_argument("--scene-token")
    args = parser.parse_args()
    print(json.dumps(evaluate_batch(args.scenario_path, args.result_dir, args.scene_token), ensure_ascii=False))


if __name__ == "__main__":
    main()
