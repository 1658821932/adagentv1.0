"""Agent-side bridge to the raw nuPlan map facts.

The agent environment has no GIS libraries, so the simulator interpreter runs a small
script that reuses tool01's map lookup and CRS helpers. Any failure is reported as
unknown evidence, never as a structural verdict.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("mapfacts_route.py")
TIMEOUT_S = 180
UNKNOWN = dict(road="unknown", lanes="unknown", space="unknown")


def python_path():
    candidate = PROJECT_ROOT / "environments/simengine/bin/python"
    return str(candidate) if candidate.is_file() else None


def unavailable(reason):
    return dict(available=False, checks=dict(UNKNOWN), facts=None,
                reasons=[reason], evidence_limits=[reason], retrieval_suggestion=None)


def collect(frames, attack):
    """Measured road/lane/space facts for the scene's own map, or unknown."""
    map_location = next((str(frame.get("map_location")) for frame in frames if frame.get("map_location")), "")
    if not map_location:
        return unavailable("Scene frames carry no map_location; the raw map cannot be located.")
    interpreter = python_path()
    if interpreter is None:
        return unavailable("Simulator interpreter is not available; the raw map cannot be read.")
    request = json.dumps(dict(frames=frames, map_location=map_location, attack=attack), ensure_ascii=False)
    try:
        completed = subprocess.run([interpreter, str(SCRIPT)], input=request, capture_output=True,
                                   text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        return unavailable("Raw map reader failed (" + type(error).__name__ + "); structure stays unknown.")
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()[-1:] or [""]
        return unavailable("Raw map reader exited with " + str(completed.returncode) + ": " + detail[0][:200])
    try:
        payload = json.loads(completed.stdout)
    except ValueError:
        return unavailable("Raw map reader returned no usable JSON; structure stays unknown.")
    if not isinstance(payload, dict) or not payload.get("available"):
        reason = payload.get("reason") if isinstance(payload, dict) else None
        return unavailable(str(reason or "Raw map data is unavailable; structure stays unknown."))
    checks = payload.get("checks") or dict(UNKNOWN)
    return dict(available=True, checks={name: checks.get(name, "unknown") for name in UNKNOWN},
                facts=payload.get("facts"), reasons=list(payload.get("reasons") or []),
                evidence_limits=list(payload.get("evidence_limits") or []),
                retrieval_suggestion=payload.get("retrieval_suggestion"))
