"""Deterministic local mock of the failure analyzer.

It returns the same result contract as the real analysis with fixed values and never
touches an archive, the map or a model.
"""

MOCK_MAP_LOCATION = "us-pa-pittsburgh-hazelwood"
MOCK_EVIDENCE_DIR = "mock://tool01/evidence"
MOCK_FAILURE = {
    "outcome": "collision",
    "interaction": "crossing",
    "ego_cause": "在丁字路口右转时，没有对从右侧支路左转过来的车辆让行",
}
MOCK_ATTACK = {
    "map_context": "t_junction",
    "ego_action": "turn_right",
    "other_origin": "right_arm",
    "other_action": "turn_left",
}
MOCK_ENVIRONMENT = {
    "time": "2021-09-16T20:16:41+00:00",
    "weather": "clear",
    "extra": "mock scene: dry road at a t-junction with a right arm",
}
MOCK_VERIFICATION = {"matched": True, "notes": []}


def run(input_data: dict) -> dict:
    """Return a deterministic mock failure analysis in the real output shape."""
    token = input_data["token"]
    return {
        "status": "success",
        "tool": "failure_analyzer",
        "result": {
            "token": token,
            "simulation": input_data.get("simulation"),
            "map_location": MOCK_MAP_LOCATION,
            "evidence_dir": MOCK_EVIDENCE_DIR,
            "failure": dict(MOCK_FAILURE),
            "attack": dict(MOCK_ATTACK),
            "environment": dict(MOCK_ENVIRONMENT),
            "verification": dict(MOCK_VERIFICATION),
        },
    }