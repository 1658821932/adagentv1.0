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
    tmp = file_path.with_name(file_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, file_path)
    return payload["used_scene_tokens"]


def extract_succeeded_base_scene_token(run_dir: str | Path, *, max_pdm_score: float = 0.5, min_violation_step: int = 6) -> str | None:
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
            if isinstance(score, (int, float)) and float(score) < max_pdm_score and step_ok:
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
