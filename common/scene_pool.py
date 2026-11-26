"""Batch-scoped base-scene exclusion pool for Tool 03 (video_retriever).

Only base scenes that produced a verified successful edit (Tool 05 completed) are
recorded into a batch exclusion file; Tool 04 unsuitable verdicts or Tool 05 failures
remain scoped to the single run via adapter/client.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

EXCLUDE_ENV_VAR = "ADAGENT_EXCLUDE_SCENE_TOKENS_FILE"
PDM_THRESHOLD_ENV_VAR = "ADAGENT_PDM_FAILURE_THRESHOLD"
DEFAULT_PDM_FAILURE_THRESHOLD = 0.5


def get_pdm_failure_threshold(override: float | None = None) -> float:
    """Return configurable PDM failure threshold (default 0.5, overridable via ADAGENT_PDM_FAILURE_THRESHOLD)."""
    if override is not None:
        val = float(override)
        if not (0.0 < val <= 1.0):
            raise ValueError(f"pdm_threshold must be in (0, 1], got {val}")
        return val
    raw = (os.getenv(PDM_THRESHOLD_ENV_VAR) or "").strip()
    if raw:
        try:
            val = float(raw)
            if 0.0 < val <= 1.0:
                return val
        except ValueError:
            pass
    return DEFAULT_PDM_FAILURE_THRESHOLD


def normalize_scene_token(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text.split("-")[-1]


def load_excluded_scene_tokens(path: str | Path | None) -> list[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.is_file():
        return []
    raw = file_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    tokens: list[Any] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            tokens = parsed
        elif isinstance(parsed, dict):
            for key in ("used_scene_tokens", "exclude_scene_tokens", "excluded_scene_tokens", "tokens"):
                val = parsed.get(key)
                if isinstance(val, list):
                    tokens = val
                    break
    except json.JSONDecodeError:
        tokens = [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]
    normalized = {tok for item in tokens if (tok := normalize_scene_token(item))}
    return sorted(normalized)


def record_used_scene_token(
    path: str | Path,
    scene_token: str,
    *,
    source_token: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Atomically record a successfully used base scene token into the batch exclusion file."""
    token = normalize_scene_token(scene_token)
    if not token:
        raise ValueError("scene_token must be a non-empty string")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = file_path.with_name(file_path.name + ".lock")
    lock_fd = None
    try:
        try:
            import fcntl
            lock_fd = open(lock_path, "a+", encoding="utf-8")
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        except OSError:
            lock_fd = None

        existing_tokens = set(load_excluded_scene_tokens(file_path))
        records: list[dict[str, Any]] = []
        if file_path.is_file():
            try:
                parsed = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
                    records = list(parsed["records"])
            except json.JSONDecodeError:
                pass

        existing_tokens.add(token)
        entry: dict[str, Any] = {"scene_token": token}
        if source_token:
            entry["source_token"] = str(source_token)
        if metadata:
            entry.update(metadata)
        records.append(entry)

        payload = {
            "used_scene_tokens": sorted(existing_tokens),
            "records": records,
        }
        tmp = file_path.with_name(f"{file_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, file_path)
        return payload["used_scene_tokens"]
    finally:
        if lock_fd is not None:
            try:
                import fcntl
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_fd.close()


def extract_succeeded_base_scene_token(run_dir: str | Path, *, max_pdm_score: float | None = None, min_violation_step: int = 6) -> str | None:
    threshold = get_pdm_failure_threshold(max_pdm_score)
    """Return the base scene_token ONLY if Tool 05 succeeded AND (if Tool 06 ran) closed-loop PDM < max_pdm_score."""
    tools_dir = Path(run_dir) / "tools"
    if not tools_dir.is_dir():
        return None
    # If policy_runner (Tool 06) ran in this run, require at least one evaluation with score < max_pdm_score
    t06_dirs = sorted(tools_dir.glob("step_*_policy_runner"))
    if t06_dirs:
        passed_failure_gate = False
        for t06_dir in reversed(t06_dirs):
            out_p = t06_dir / "output.json"
            if not out_p.is_file():
                continue
            try:
                out = json.loads(out_p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if out.get("status") != "success":
                continue
            res = out.get("result") or {}
            sub = res.get("pdm_subscores") or res.get("metrics") or {}
            score = sub.get("score")
            fvs = sub.get("first_violation_step")
            step_ok = (fvs is None) or (isinstance(fvs, (int, float)) and float(fvs) >= min_violation_step)
            if isinstance(score, (int, float)) and float(score) < threshold and step_ok:
                passed_failure_gate = True
                break
        if not passed_failure_gate:
            return None
    step_dirs = sorted(tools_dir.glob("step_*_attack_vehicle_editor"))
    for step_dir in reversed(step_dirs):
        out_path = step_dir / "output.json"
        if not out_path.is_file():
            continue
        try:
            output = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if output.get("status") != "success":
            continue
        result = output.get("result") or {}
        if result.get("status") != "completed":
            continue
        # Extract base scene_token from input.json scene_ref or result.scene_id
        in_path = step_dir / "input.json"
        if in_path.is_file():
            try:
                inp = json.loads(in_path.read_text(encoding="utf-8"))
                ref = inp.get("scene_ref")
                if isinstance(ref, dict):
                    tok = normalize_scene_token(ref.get("scene_token") or ref.get("scene_id"))
                    if tok:
                        return tok
            except (OSError, json.JSONDecodeError):
                pass
        tok = normalize_scene_token(result.get("scene_token") or result.get("scene_id"))
        if tok:
            return tok
    return None


def record_run_if_configured(run_dir: str | Path, *, run_id: str | None = None) -> str | None:
    """If ADAGENT_EXCLUDE_SCENE_TOKENS_FILE is set and run produced a verified failure scene, record its base token."""
    pool_path = (os.getenv(EXCLUDE_ENV_VAR) or "").strip()
    if not pool_path:
        return None
    base_token = extract_succeeded_base_scene_token(run_dir)
    if not base_token:
        return None
    meta = {"run_dir": str(Path(run_dir).resolve())}
    if run_id:
        meta["run_id"] = str(run_id)
    record_used_scene_token(pool_path, base_token, metadata=meta)
    return base_token


def classify_run_result(
    run_dir: str | Path,
    *,
    pdm_threshold: float | None = None,
    min_violation_step: int = 6,
) -> dict[str, Any]:
    """Classify a completed/terminated agent run into the 3-bucket batch strategy.

    - class_1_success: Tool 06 produced PDM < pdm_threshold AND first_violation_step >= min_violation_step.
    - class_2_retry_speed: Tool 06 produced a valid numeric PDM score, but PDM >= pdm_threshold (or violated too early < min_violation_step).
    - class_3_no_result: No valid Tool 06 score (e.g. Tool 05 timed out >15m, unable_to_complete, or upstream error).
    """
    threshold = get_pdm_failure_threshold(pdm_threshold)
    rdir = Path(run_dir)
    tools_dir = rdir / "tools"
    if not tools_dir.is_dir():
        return {
            "bucket": "class_3_no_result",
            "pdm_score": None,
            "first_violation_step": None,
            "base_scene_token": None,
            "tool05_status": None,
            "threshold": threshold,
            "reason": "no_tools_output",
        }

    # Inspect latest Tool 05 status and base scene token
    tool05_status = None
    last_base_token = None
    for step_dir in reversed(sorted(tools_dir.glob("step_*_attack_vehicle_editor"))):
        out_p = step_dir / "output.json"
        if out_p.is_file():
            try:
                out = json.loads(out_p.read_text(encoding="utf-8"))
                if out.get("status") == "success":
                    tool05_status = str((out.get("result") or {}).get("status") or "unknown")
                else:
                    err = out.get("error") or {}
                    err_msg = str(err.get("message") or err.get("type") or out.get("status") or "error")
                    tool05_status = "timeout" if "timeout" in err_msg.lower() else "error"
            except (OSError, json.JSONDecodeError):
                tool05_status = "error"
        in_p = step_dir / "input.json"
        if in_p.is_file():
            try:
                inp = json.loads(in_p.read_text(encoding="utf-8"))
                ref = inp.get("scene_ref")
                if isinstance(ref, dict):
                    last_base_token = normalize_scene_token(ref.get("scene_token") or ref.get("scene_id"))
            except (OSError, json.JSONDecodeError):
                pass
        break

    # Inspect Tool 06 runs (prefer any passing class_1 run; else latest numeric score -> class_2)
    best_class2 = None
    for t06_dir in reversed(sorted(tools_dir.glob("step_*_policy_runner"))):
        out_p = t06_dir / "output.json"
        if not out_p.is_file():
            continue
        try:
            out = json.loads(out_p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if out.get("status") != "success":
            continue
        res = out.get("result") or {}
        sub = res.get("pdm_subscores") or res.get("metrics") or {}
        score = sub.get("score")
        if not isinstance(score, (int, float)):
            continue
        fvs = sub.get("first_violation_step")
        step_ok = (fvs is None) or (isinstance(fvs, (int, float)) and float(fvs) >= min_violation_step)
        if float(score) < threshold and step_ok:
            return {
                "bucket": "class_1_success",
                "pdm_score": float(score),
                "first_violation_step": float(fvs) if isinstance(fvs, (int, float)) else None,
                "base_scene_token": extract_succeeded_base_scene_token(rdir, max_pdm_score=threshold, min_violation_step=min_violation_step) or last_base_token,
                "tool05_status": tool05_status,
                "threshold": threshold,
                "reason": f"pdm={float(score):.4f} < {threshold}",
            }
        if best_class2 is None:
            reason = (
                f"pdm={float(score):.4f} >= {threshold}"
                if float(score) >= threshold
                else f"pdm={float(score):.4f} < {threshold} but first_violation_step={fvs} < {min_violation_step}"
            )
            best_class2 = {
                "bucket": "class_2_retry_speed",
                "pdm_score": float(score),
                "first_violation_step": float(fvs) if isinstance(fvs, (int, float)) else None,
                "base_scene_token": last_base_token,
                "tool05_status": tool05_status,
                "threshold": threshold,
                "reason": reason,
            }

    if best_class2 is not None:
        return best_class2

    return {
        "bucket": "class_3_no_result",
        "pdm_score": None,
        "first_violation_step": None,
        "base_scene_token": last_base_token,
        "tool05_status": tool05_status,
        "threshold": threshold,
        "reason": f"no_tool06_score (tool05_status={tool05_status})",
    }
