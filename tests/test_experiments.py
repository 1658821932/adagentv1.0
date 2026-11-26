"""Tests for unified outputs/experiments/ iteration & ablation layout."""
from __future__ import annotations

import json
import pytest

from common.experiments import (
    EXPERIMENT_ENV_VAR,
    active_experiment_dir,
    init_experiment,
    list_experiments,
    resolve_experiment_paths,
    set_active_experiment,
)


def test_experiment_lifecycle_and_active_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv(EXPERIMENT_ENV_VAR, raising=False)
    e0 = init_experiment("iter_00_base", parent=None, kind="iteration", metrics={"PDMS": 0.57555}, root=tmp_path)
    (e0 / "checkpoint.pth").write_bytes(b"ckpt0")
    (e0 / "eval_288").mkdir()

    e1 = init_experiment("iter_01_syn79", parent="iter_00_base", kind="iteration", root=tmp_path)
    (e1 / "checkpoint.pth").write_bytes(b"ckpt1")

    ea = init_experiment("ablation_79base", parent="iter_00_base", kind="ablation", root=tmp_path)
    assert not (ea / "base_sim_cache").exists()
    assert (e1 / "base_sim_cache").is_dir()

    set_active_experiment("iter_01_syn79", root=tmp_path)
    assert active_experiment_dir(root=tmp_path) == e1.resolve()

    resolved = resolve_experiment_paths(root=tmp_path)
    assert resolved["exp_id"] == "iter_01_syn79"
    assert resolved["parent"] == "iter_00_base"
    assert resolved["checkpoint"] == str((e1 / "checkpoint.pth").resolve())

    # Env var overrides active symlink
    monkeypatch.setenv(EXPERIMENT_ENV_VAR, "iter_00_base")
    assert active_experiment_dir(root=tmp_path) == e0.resolve()

    rows = list_experiments(root=tmp_path)
    assert [r["exp_id"] for r in rows] == ["ablation_79base", "iter_00_base", "iter_01_syn79"]
    assert next(r for r in rows if r["exp_id"] == "iter_00_base")["active"] is True


def test_invalid_experiment_id_rejected(tmp_path):
    for bad in ("", "active", "../escape", "a/b"):
        with pytest.raises(ValueError):
            init_experiment(bad, root=tmp_path)
