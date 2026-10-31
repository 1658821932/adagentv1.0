"""Verify batch failure filtering matches trajectories by scene identity.

Covers out-of-order metadata, multi-scene batches, and missing/ambiguous
identity handling.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_servers.worldengine.adapter.evaluate_batch import evaluate_batch


def _scene(token, ego_positions, other_positions=None, other_name="agent_0"):
    other = {
        "position": other_positions if other_positions is not None else [[1000.0, 1000.0, 0.0]] * len(ego_positions),
        "valid": [True] * len(ego_positions),
        "length": [4.5] * len(ego_positions),
        "width": [2.0] * len(ego_positions),
    }
    return {
        "id": token,
        "scene_token": token,
        "metadata": {"rear_axle_to_center_dist": 0.0},
        "object_track": {
            "ego": {"state": {
                "position": ego_positions,
                "valid": [True] * len(ego_positions),
                "length": [4.5] * len(ego_positions),
                "width": [2.0] * len(ego_positions),
            }},
            other_name: {"state": other},
        },
    }


def _frames(token, log_name, positions):
    return [
        {
            "frame_idx": i,
            "scene_token": token,
            "log_name": log_name,
            "ego2global_translation": [p[0], p[1], p[2]],
            "ego2global": [[1.0, 0.0, 0.0, p[0]], [0.0, 1.0, 0.0, p[1]], [0.0, 0.0, 1.0, p[2]], [0.0, 0.0, 0.0, 1.0]],
        }
        for i, p in enumerate(positions)
    ]


def _write_metadata(meta_dir, name, frames):
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / name).open("wb") as fh:
        pickle.dump(frames, fh)


def _write_report(result_dir, names):
    import json
    with (Path(result_dir) / "runner_report_0.json").open("w") as fh:
        json.dump([{"scenario_name": n} for n in names], fh)


def test_multi_scene_matched_by_identity_despite_order(tmp_path):
    """Two scenes whose metadata/report order is reversed relative to the pkl.

    Only scene A has an obstacle on its path, so only A must be flagged,
    regardless of file ordering.
    """
    token_a = "log-a-aaaaaaaaaaaaaaaa"
    token_b = "log-b-bbbbbbbbbbbbbbbb"
    scenario = tmp_path / "scenarios.pkl"
    with scenario.open("wb") as fh:
        pickle.dump({
            token_a: _scene(token_a, [[0, 0, 0], [1, 0, 0]], other_positions=[[0.5, 0, 0], [0.5, 0, 0]]),
            token_b: _scene(token_b, [[0, 0, 0], [1, 0, 0]]),
        }, fh)

    result_dir = tmp_path / "result"
    meta_dir = result_dir / "WE_output/openscene_format/meta_datas"
    # B's metadata is written first (reverse alphabetical/order-of-pkl),
    # and the runner report lists B before A.
    _write_metadata(meta_dir, f"{token_b}.pkl", _frames(token_b, "log-b", [[0, 0, 0], [1, 0, 0]]))
    _write_metadata(meta_dir, f"{token_a}.pkl", _frames(token_a, "log-a", [[0, 0, 0], [1, 0, 0]]))
    _write_report(result_dir, [token_b, token_a])

    out = evaluate_batch(scenario, result_dir)
    assert out["scene_count"] == 2
    by_id = {row["scene_id"]: row for row in out["scene_results"]}
    assert by_id[token_a]["collision"] is True
    assert by_id[token_b]["collision"] is False
    assert by_id[token_a]["trajectory_source"] == "worldengine_openscene_metadata"
    assert out["is_failure_case"] is True
    assert [f["scene_id"] for f in out["failure_cases"]] == [token_a]
    assert out["skipped_scene_ids"] == []


def test_combined_metadata_file_split_by_content(tmp_path):
    """A single combined metadata file holds frames for both scenes."""
    token_a = "log-a-aaaaaaaaaaaaaaaa"
    token_b = "log-b-bbbbbbbbbbbbbbbb"
    scenario = tmp_path / "scenarios.pkl"
    with scenario.open("wb") as fh:
        pickle.dump({
            token_a: _scene(token_a, [[0, 0, 0]], other_positions=[[0, 0, 0]]),
            token_b: _scene(token_b, [[0, 0, 0]]),
        }, fh)
    result_dir = tmp_path / "result"
    meta_dir = result_dir / "WE_output/openscene_format/meta_datas"
    combined = _frames(token_b, "log-b", [[0, 0, 0]]) + _frames(token_a, "log-a", [[0, 0, 0]])
    _write_metadata(meta_dir, "combined.pkl", combined)
    _write_report(result_dir, [token_a, token_b])

    out = evaluate_batch(scenario, result_dir)
    by_id = {row["scene_id"]: row for row in out["scene_results"]}
    assert by_id[token_a]["collision"] is True
    assert by_id[token_b]["collision"] is False


def test_conflicting_filename_and_content_rejected(tmp_path):
    """A file named after A but containing B's frames must fail loudly."""
    token_a = "log-a-aaaaaaaaaaaaaaaa"
    token_b = "log-b-bbbbbbbbbbbbbbbb"
    scenario = tmp_path / "scenarios.pkl"
    with scenario.open("wb") as fh:
        pickle.dump({token_a: _scene(token_a, [[0, 0, 0]]),
                     token_b: _scene(token_b, [[0, 0, 0]])}, fh)
    result_dir = tmp_path / "result"
    meta_dir = result_dir / "WE_output/openscene_format/meta_datas"
    _write_metadata(meta_dir, f"{token_a}.pkl", _frames(token_b, "log-b", [[0, 0, 0]]))
    with pytest.raises(ValueError, match="Ambiguous or conflicting"):
        evaluate_batch(scenario, result_dir)


def test_scene_token_suffix_selects_single_scene(tmp_path):
    token_a = "log-a-aaaaaaaaaaaaaaaa"
    token_b = "log-b-bbbbbbbbbbbbbbbb"
    scenario = tmp_path / "scenarios.pkl"
    with scenario.open("wb") as fh:
        pickle.dump({
            token_a: _scene(token_a, [[0, 0, 0]], other_positions=[[0, 0, 0]]),
            token_b: _scene(token_b, [[0, 0, 0]]),
        }, fh)
    out = evaluate_batch(scenario, scene_token="aaaaaaaaaaaaaaaa")
    assert out["evaluated_scene_ids"] == [token_a]
    assert out["skipped_scene_ids"] == [token_b]
    assert out["collision"] is True


def test_missing_metadata_fails_explicitly(tmp_path):
    token_a = "log-a-aaaaaaaaaaaaaaaa"
    token_b = "log-b-bbbbbbbbbbbbbbbb"
    scenario = tmp_path / "scenarios.pkl"
    with scenario.open("wb") as fh:
        pickle.dump({token_a: _scene(token_a, [[0, 0, 0]]),
                     token_b: _scene(token_b, [[0, 0, 0]])}, fh)
    result_dir = tmp_path / "result"
    meta_dir = result_dir / "WE_output/openscene_format/meta_datas"
    _write_metadata(meta_dir, f"{token_a}.pkl", _frames(token_a, "log-a", [[0, 0, 0]]))
    _write_report(result_dir, [token_a, token_b])
    with pytest.raises(ValueError, match="Missing executed metadata"):
        evaluate_batch(scenario, result_dir)


def test_unmatchable_metadata_fails(tmp_path):
    token_a = "log-a-aaaaaaaaaaaaaaaa"
    scenario = tmp_path / "scenarios.pkl"
    with scenario.open("wb") as fh:
        pickle.dump({token_a: _scene(token_a, [[0, 0, 0]])}, fh)
    result_dir = tmp_path / "result"
    meta_dir = result_dir / "WE_output/openscene_format/meta_datas"
    _write_metadata(meta_dir, "mystery.pkl", _frames("unknown-token", "unknown-log", [[0, 0, 0]]))
    with pytest.raises(ValueError, match="Cannot match"):
        evaluate_batch(scenario, result_dir)


def test_non_contiguous_frames_rejected(tmp_path):
    token_a = "log-a-aaaaaaaaaaaaaaaa"
    scenario = tmp_path / "scenarios.pkl"
    with scenario.open("wb") as fh:
        pickle.dump({token_a: _scene(token_a, [[0, 0, 0], [1, 0, 0]])}, fh)
    result_dir = tmp_path / "result"
    meta_dir = result_dir / "WE_output/openscene_format/meta_datas"
    frames = _frames(token_a, "log-a", [[0, 0, 0], [1, 0, 0]])
    frames[1]["frame_idx"] = 5  # gap
    _write_metadata(meta_dir, f"{token_a}.pkl", frames)
    _write_report(result_dir, [token_a])
    with pytest.raises(ValueError, match="contiguous"):
        evaluate_batch(scenario, result_dir)
