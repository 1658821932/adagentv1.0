"""Real failure analyzer backed by worldengine_failure_analysis (wfa).

Replaces the legacy pmagent free-form summarizer with wfa's evidence-based
VLM forensic pipeline: official closed-loop metrics + image forensics +
numeric verification, keyed by WorldEngine scene token.
"""

from __future__ import annotations

from typing import Any

from mcp_servers.worldengine_failure_analysis.adapter.service import analyze_scene, WfaError


def _map_to_failure_analysis(result) -> dict[str, Any]:
    """Map a wfa scene result onto the compact downstream failure_analysis contract.

    Only what downstream needs is returned; full forensic detail
    (candidate assessments, image/evidence inventory, report) stays on disk and is
    reachable through ``result_path``.
    """
    return {
        "failure_type": [result.reason_category or "unknown"],
        "failure_reason": result.failure_reason,
        "key_scene_factors": list(result.observations or [])[:5],
        "official_metrics": dict(result.official_metrics or {}),
        "uncertainty": result.uncertainty or "",
    }


def run(input_data: dict[str, Any]) -> dict[str, Any]:
    token = (
        input_data.get("token")
        or input_data.get("scene_token")
        or (input_data.get("failure_case") or {}).get("token")
    )
    if not token:
        raise ValueError(
            "failure_analyzer (wfa) requires a scene 'token' from a WorldEngine "
            "closed-loop run. For free-form failure_case lists, run a WorldEngine "
            "policy first (policy_runner -> failure_case_filter) to obtain tokens."
        )
    try:
        result = analyze_scene(
            str(token),
            simulation=(input_data.get("simulation") or input_data.get("we_simulation")
                        or input_data.get("simulation_record") or input_data.get("result_root")),
            timeout=int(input_data.get("timeout", 300)),
        )
    except WfaError as error:
        # Old adapters/plain exceptions may not carry the extended metadata.
        # Error reporting must never replace the backend cause with AttributeError.
        details = getattr(error, 'details', {})
        if not isinstance(details, dict):
            details = {}
        return dict(status='error', tool='failure_analyzer',
                    error=dict(type='backend_environment_error' if details.get('failure_stage') == 'environment'
                               else 'failure_analysis_error', message=str(error), **details))
    return {
        "status": "success",
        "tool": "failure_analyzer",
        "result": {
            "failure_analysis": _map_to_failure_analysis(result),
            "token": result.token,
            "simulation": (result.simulation or {}).get("record"),
            "evidence_dir": result.result_path,
        },
    }

