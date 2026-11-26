"""Unit tests for configurable PDM threshold, 3-bucket classification, and Round 2 ordering."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_run(run_dir: Path, *, base_tok: str, t05_status: str = "completed", t06_score=None, t06_step=None):
    t05 = run_dir / "tools" / "step_01_attack_vehicle_editor"
    t05.mkdir(parents=True, exist_ok=True)
    (t05 / "input.json").write_text(json.dumps({"scene_ref": {"scene_token": base_tok}}))
    if t05_status in ("completed", "unable_to_complete"):
        (t05 / "output.json").write_text(json.dumps({"status": "success", "result": {"status": t05_status}}))
    else:
        (t05 / "output.json").write_text(json.dumps({"status": "error", "error": {"type": t05_status, "message": t05_status}}))

    if t06_score is not None:
        t06 = run_dir / "tools" / "step_02_policy_runner"
        t06.mkdir(parents=True, exist_ok=True)
        sub = {"score": t06_score}
        if t06_step is not None:
            sub["first_violation_step"] = t06_step
        (t06 / "output.json").write_text(json.dumps({"status": "success", "result": {"pdm_subscores": sub}}))


def test_configurable_pdm_threshold(monkeypatch):
    from common.scene_pool import PDM_THRESHOLD_ENV_VAR, get_pdm_failure_threshold

    monkeypatch.delenv(PDM_THRESHOLD_ENV_VAR, raising=False)
    assert get_pdm_failure_threshold() == 0.5

    monkeypatch.setenv(PDM_THRESHOLD_ENV_VAR, "0.3")
    assert get_pdm_failure_threshold() == 0.3

    assert get_pdm_failure_threshold(0.8) == 0.8


def test_classify_three_buckets_and_variable_threshold(tmp_path, monkeypatch):
    from common.scene_pool import PDM_THRESHOLD_ENV_VAR, classify_run_result

    monkeypatch.setenv(PDM_THRESHOLD_ENV_VAR, "0.5")

    r_ok = tmp_path / "r_ok"
    _write_run(r_ok, base_tok="base_a", t06_score=0.4, t06_step=8)
    v_ok = classify_run_result(r_ok)
    assert v_ok["bucket"] == "class_1_success"
    assert v_ok["base_scene_token"] == "base_a"

    # Lowering threshold to 0.3 flips score=0.4 from Class 1 to Class 2!
    v_strict = classify_run_result(r_ok, pdm_threshold=0.3)
    assert v_strict["bucket"] == "class_2_retry_speed"

    # Early violation (< step 6) -> Class 2
    r_early = tmp_path / "r_early"
    _write_run(r_early, base_tok="base_b", t06_score=0.0, t06_step=4)
    assert classify_run_result(r_early)["bucket"] == "class_2_retry_speed"

    # Tool 05 timeout / no Tool 06 -> Class 3
    r_timeout = tmp_path / "r_timeout"
    _write_run(r_timeout, base_tok="base_c", t05_status="TimeoutExpired", t06_score=None)
    v_to = classify_run_result(r_timeout)
    assert v_to["bucket"] == "class_3_no_result"
    assert v_to["tool05_status"] == "timeout"


def test_round2_runs_class2_before_class3():
    from scripts.run_batch_generation import round2_queue

    tokens = ["t1", "t2", "t3", "t4", "t5"]
    state = {
        "items": {
            "t1": {"round1": {"bucket": "class_3_no_result"}},
            "t2": {"round1": {"bucket": "class_1_success"}},
            "t3": {"round1": {"bucket": "class_2_retry_speed"}},
            "t4": {"round1": {"bucket": "class_3_no_result"}},
            "t5": {"round1": {"bucket": "class_2_retry_speed"}},
        }
    }
    q = round2_queue(state, tokens)
    assert q == [
        ("t3", "class_2_retry_speed"),
        ("t5", "class_2_retry_speed"),
        ("t1", "class_3_no_result"),
        ("t4", "class_3_no_result"),
    ]
