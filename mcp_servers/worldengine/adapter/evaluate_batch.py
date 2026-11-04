"""Filter executed scenes exclusively by saved official closed-loop PDM scores."""
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
    matched = defaultdict(list)
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
    return matched


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


def evaluate_official_pdm(scenario_path, result_dir, index, selected):
    """Select PDM-zero scenes using saved scores and matched execution metadata."""
    import csv
    root = Path(result_dir)
    candidates = sorted((root / "WE_output/openscene_format").glob("all_scenes_pdm_averages_*.csv"))
    if not candidates:
        candidates = sorted(root.glob("all_scenes_pdm_averages_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No official PDM score file in {root}; run policy_runner with PDM evaluation first")
    if len(candidates) != 1:
        raise ValueError("Expected one official PDM score file")
    score_file = candidates[0]
    rows = {}
    with score_file.open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = (row.get("token") or "").strip()
            if token == "overall_average":
                continue
            identity = resolve([token], index, "PDM scene token")
            if identity in rows:
                raise ValueError("Duplicate official PDM scene: " + identity)
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Missing or invalid official PDM score") from exc
            if not math.isfinite(score) or score < 0:
                raise ValueError("Non-finite or negative official PDM score")
            rows[identity] = score
    if not rows:
        raise ValueError("Official PDM score file has no scene rows")
    selected = (reported_scenes(root, index) or set(rows)) if selected is None else selected
    missing = selected - set(rows)
    if missing:
        raise ValueError("Missing official PDM scores: " + ", ".join(sorted(missing)))
    meta_dir = score_file.parent / "meta_datas"
    matched = metadata_by_scene(meta_dir, index)
    missing = selected - set(matched)
    if missing:
        raise ValueError("Missing executed metadata: " + ", ".join(sorted(missing)))
    results, failures = [], []
    for identity in sorted(selected):
        indices = [frame.get("frame_idx") for frame in matched[identity]]
        if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) for i in indices):
            raise ValueError(f"{identity}: frame_idx must be an integer")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"{identity}: metadata frames must be unique and contiguous from zero")
        score = rows[identity]
        evaluation = {"scene_id": identity, "pdm_score": score}
        results.append(evaluation)
        if score == 0.0:
            failures.append({"scene_id": identity, "scenario_pkl": str(scenario_path), "pdm_score": score})
    return {"is_failure_case": bool(failures), "failure_cases": failures,
        "scene_count": len(results), "scene_results": results,
        "pdm_scores_file": str(score_file)}


def evaluate_batch(scenario_path, result_dir=None, scene_token=None):
    if not isinstance(result_dir, (str, Path)) or not str(result_dir).strip():
        raise ValueError("result_dir with official PDM scores is required")
    if __package__:
        from .prepare_scenarios import SceneDataUnpickler
    else:
        from prepare_scenarios import SceneDataUnpickler
    with Path(scenario_path).open("rb") as stream:
        _, index = scenes_from(SceneDataUnpickler(stream).load())
    selected = {resolve([scene_token], index, "requested scene_token")} if scene_token else None
    return evaluate_official_pdm(scenario_path, result_dir, index, selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_path")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--scene-token")
    args = parser.parse_args()
    print(json.dumps(evaluate_batch(args.scenario_path, args.result_dir, args.scene_token), ensure_ascii=False))


if __name__ == "__main__":
    main()
