"""Tests for policy closed-loop simulation ego trajectory reference in Tool 04 & Tool 05."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from common.baseline_sim import (
    PROJECT_ROOT,
    reference_ego_path_from_scene,
    resolve_baseline_frames_path,
)


def _make_frame(ts_us: int, x: float, y: float, yaw: float = 0.0) -> dict:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return {
        "timestamp": ts_us,
        "ego2global": [
            [c, -s, 0.0, x],
            [s,  c, 0.0, y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def test_reference_ego_uses_policy_simulation_and_extrapolates_tail(tmp_path):
    cache_base = tmp_path / "cache"
    tok = "aaaa1111bbbb2222"
    sim_dir = cache_base / "e2e_vadv2_50pct_ep8" / tok
    sim_dir.mkdir(parents=True)
    # Recorded replay moves at 10 m/s for 6 frames (0..2.5s)
    rec = [_make_frame(1_000_000 + i * 500_000, 100.0 + i * 5.0, 20.0) for i in range(6)]
    # Policy sim moves at 4 m/s for 4 frames (0..1.5s)
    sim = [_make_frame(1_000_000 + i * 500_000, 100.0 + i * 2.0, 20.0) for i in range(4)]
    (sim_dir / "baseline_frames.json").write_text(json.dumps(sim))

    scene_dir = tmp_path / "samples" / tok
    scene_dir.mkdir(parents=True)
    times, ego_xy, ego_yaw, meta = reference_ego_path_from_scene(
        scene_dir, rec, cache_base=cache_base
    )
    assert meta["source"] == "policy_simulation"
    assert meta["sim_frames"] == 4
    assert times.shape == (6,)
    assert ego_xy.shape == (6, 2)
    # Frame 0..3 match sim (0, 2, 4, 6m); frames 4..5 extrapolate at 4 m/s (8, 10m)
    assert np.allclose(ego_xy[:, 0], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], atol=1e-5)
    assert np.allclose(ego_xy[:, 1], 0.0, atol=1e-5)
    assert np.allclose(ego_yaw, 0.0, atol=1e-5)


def test_reference_ego_falls_back_to_recorded_when_no_cache(tmp_path):
    rec = [_make_frame(1_000_000 + i * 500_000, 50.0 + i * 3.0, 10.0) for i in range(5)]
    times, ego_xy, ego_yaw, meta = reference_ego_path_from_scene(
        tmp_path / "uncached_tok", rec, cache_base=tmp_path / "empty_cache"
    )
    assert meta["source"] == "recorded_replay"
    assert np.allclose(ego_xy[:, 0], [0.0, 3.0, 6.0, 9.0, 12.0], atol=1e-5)


def test_real_navtrain_baseline_cache_integration():
    tok = "0090491b97185efb"
    bp = resolve_baseline_frames_path(tok)
    sample_dir = PROJECT_ROOT / "data/navtrain_ep_per1/samples" / tok
    if bp is None or not (sample_dir / "metadata/frames.json").is_file():
        pytest.skip("local navtrain baseline cache not mounted")
    rec = json.loads((sample_dir / "metadata/frames.json").read_text())
    times, ego_xy, ego_yaw, meta = reference_ego_path_from_scene(sample_dir, rec)
    assert meta["source"] == "policy_simulation"
    assert meta["sim_frames"] == 16
    assert len(times) == len(rec)
    # Verify sim trajectory genuinely differs from recorded replay at frame 15
    rec_world = np.array([np.array(f["ego2global"])[:2, 3] for f in rec])
    sim_frames = json.loads(bp.read_text())
    sim_world = np.array([np.array(f["ego2global"])[:2, 3] for f in sim_frames])
    assert float(np.linalg.norm(sim_world[15] - rec_world[15])) > 5.0
