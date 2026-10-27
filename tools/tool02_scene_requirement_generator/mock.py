"""Deterministic mock; validated against the same contract as real."""
from .real import TOOL


def _result(data):
    analysis = data.failure_analysis
    kind = analysis.get("failure_type", "")
    if isinstance(kind, list):
        kind = ", ".join(str(item) for item in kind)
    reason = str(analysis.get("failure_reason") or "unspecified driving failure")
    risks = analysis.get("risk_factors", [])
    if not isinstance(risks, list):
        risks = [str(risks)]
    factors = analysis.get("key_scene_factors", [])
    if not isinstance(factors, list):
        factors = [str(factors)]
    return {
        "scene_requirement": "Create a scenario testing: " + reason,
        "failure_type": str(kind),
        "failure_reason": reason,
        "required_conditions": [str(item) for item in factors if str(item).strip()],
        "required_actors": ["ego vehicle", "relevant traffic participant"],
        "risk_factors": [str(item) for item in risks if str(item).strip()],
        "target_effect": "Evaluate the ego policy response to the specified hazard.",
        "hazard_summary": reason,
    }


def run(input_data: dict) -> dict:
    return TOOL.execute(input_data, mock_result=_result)
