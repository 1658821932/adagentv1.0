"""Real failure analyzer backed by worldengine_failure_analysis (wfa).

Replaces the legacy pmagent free-form summarizer with wfa's evidence-based
VLM forensic pipeline: official closed-loop metrics + image forensics +
numeric verification, keyed by WorldEngine scene token.
"""

from __future__ import annotations

from typing import Any

from mcp_servers.worldengine_failure_analysis.adapter.service import analyze_scene


def _map_to_failure_analysis(result) -> dict[str, Any]:
    """Map a wfa scene result onto the downstream failure_analysis contract.

    Keeps the fields the agent chain consumes (failure_type, failure_reason,
    key_scene_factors, risk_factors, supporting/uncovered case ids) while
    preserving wfa's full forensic evidence under 'forensics'.
    """
    reason_category = result.reason_category or "unknown"
    failure_type = [reason_category]
    # Derive coarse risk factors from candidate assessments + metrics.
    risk_factors: list[str] = []
    metrics = result.official_metrics or {}
    try:
        if float(metrics.get("no_at_fault_collisions", 1)) < 1:
            risk_factors.append("collision")
        if float(metrics.get("drivable_area_compliance", 1)) < 1:
            risk_factors.append("drivable_area_violation")
    except (TypeError, ValueError):
        pass
    if not risk_factors:
        risk_factors = ["driving_behavior"]

    key_scene_factors = [obs for obs in (result.observations or [])][:5]

    return {
        "failure_type": failure_type,
        "failure_reason": result.failure_reason,
        "key_scene_factors": key_scene_factors,
        "risk_factors": risk_factors,
        "supporting_case_ids": [result.token],
        "uncovered_case_ids": [],
        "forensics": {
            "token": result.token,
            "status": result.status,
            "reason_category": reason_category,
            "observations": result.observations,
            "candidate_assessment": result.candidate_assessment,
            "uncertainty": result.uncertainty,
            "official_metrics": result.official_metrics,
            "report_path": result.report_path,
            "result_path": result.result_path,
        },
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
    result = analyze_scene(
        str(token),
        config_path=input_data.get("config_path"),
        run_name=input_data.get("run_name") or "adagent_failure_analysis",
        resume=bool(input_data.get("resume", True)),
        timeout=int(input_data.get("timeout", 1800)),
    )
    return {
        "status": "success",
        "tool": "failure_analyzer",
        "result": {
            "failure_analysis": _map_to_failure_analysis(result),
            "token": result.token,
            "report_path": result.report_path,
            "result_path": result.result_path,
        },
    }

