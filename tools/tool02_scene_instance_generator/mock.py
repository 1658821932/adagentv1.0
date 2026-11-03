"""Deterministic compact requirements for offline interface checks."""
from .real import execute


def _result(data):
    analysis, hints = data.failure_analysis, data.variation_hints
    reason = analysis.get("failure_reason") or analysis["wrong_behavior"]
    road = hints.get("road_layout") or analysis.get("base_scene") or "Road context suitable for the source interaction"
    actor = hints.get("target_actor") or hints.get("actor_role") or "relevant traffic participant"
    conditions = [road] + [hints[k] for k in ("lighting", "weather", "traffic_density") if hints.get(k)]
    interaction = hints.get("actor_behavior") or analysis.get("interaction") or reason
    return {
        "scene_requirement": f"Construct a scene on {road}, with {actor}. Ego approaches the interaction region; {interaction}. Ego must assess the conflict and respond. " + ("Variation: " + "; ".join(f"{k}: {v}" for k, v in hints.items()) if hints else ""),
        "required_conditions": conditions,
        "required_actors": [actor],
        "target_effect": analysis.get("expected_behavior") or "Respond to the interaction safely with appropriate yielding, deceleration or stopping.",
    }


def run(input_data: dict) -> dict:
    return execute(input_data, mock_result=_result)
