"""Deterministic local mock of the failure analyzer."""

MOCK_MAP_LOCATION = "us-pa-pittsburgh-hazelwood"
MOCK_EVIDENCE_DIR = "mock://tool01/evidence"
MOCK_BEV_IMAGE = "mock://tool01/evidence/bev_overview.png"
MOCK_FAILURE = {
    "outcome": "collision",
    "failure_driver": "collision_with_vehicle",
    "target_track": "mock_track_01",
    "target_class": "bus",
    "map_context": "curve",
    "ego_behavior": "自车低速沿弯道行驶，末段近直行时于弯道段与对向车接触",
    "other_behavior": "正前方对向大型车左转迎面驶来，第7帧与自车接触",
    "summary": "在弯道上，自车低速行驶末段近直行，正前方对向大型车左转迎面驶来，第7帧发生车辆碰撞。",
}
MOCK_KINEMATICS = {
    "ego": {
        "speed_start_mps": 3.4,
        "speed_violation_mps": 3.3,
        "displacement_m": 15.2,
        "yaw_change_deg": -54.7,
        "tail_yaw_deg": -4.0,
        "is_stationary": False,
    },
    "other": {
        "size_m": [11.87, 2.86, 3.45],
        "is_stationary": False,
        "speed_start_mps": 9.3,
        "speed_violation_mps": 10.1,
        "displacement_m": 39.3,
        "yaw_change_deg": 85.1,
        "tail_yaw_deg": 47.9,
        "relative_heading_at_violation_deg": 166.9,
        "violation_distance_m": 0.0,
    },
    "trajectory_ref": "mock://tool01/evidence/trajectories.json",
}
MOCK_ENVIRONMENT = {
    "time": "2021-09-16T20:16:41+00:00",
    "weather": "clear",
    "extra": "mock scene: clear daytime urban road",
}
MOCK_VERIFICATION = {"sample_frames": [7], "notes": []}


def run(input_data: dict) -> dict:
    """Return a deterministic mock failure analysis in the real output shape."""
    token = input_data["token"]
    return {
        "status": "success",
        "tool": "failure_analyzer",
        "result": {
            "token": token,
            "violation_step": 7,
            "simulation": input_data.get("simulation"),
            "map_location": MOCK_MAP_LOCATION,
            "evidence_dir": MOCK_EVIDENCE_DIR,
            "bev_image": MOCK_BEV_IMAGE,
            "failure": dict(MOCK_FAILURE),
            "kinematics": {
                "ego": dict(MOCK_KINEMATICS["ego"]),
                "other": dict(MOCK_KINEMATICS["other"]),
                "trajectory_ref": MOCK_KINEMATICS["trajectory_ref"],
            },
            "environment": dict(MOCK_ENVIRONMENT),
            "verification": dict(MOCK_VERIFICATION),
        },
    }
