"""Deterministic fixture for the compact scene-requirement contract.

It is a fixed legal example of the output shape; it does not derive or claim a new
generation mechanism.
"""

RESULT = {
    "target_scene": "多云白天，丁字路口，自车右转并保持在本车道，右侧支路一辆车左转并保持在本车道。",
    "attack": {
        "map_context": "t_junction",
        "ego_action": "turn_right",
        "ego_lane_keeping": "keep",
        "other_origin": "right_arm",
        "other_action": "turn_left",
        "other_lane_keeping": "keep",
    },
    "candidates": [
        {
            "candidate_id": "cand_1",
            "base_video_requirement": "丁字路口前视清晰，右侧支路清晰且可加车。",
            "retrieval_query": "白天丁字路口，自车右转，右侧支路清晰",
            "attack_edit_strategy": "在右侧支路新增一辆左转车辆并设定其通过时机。",
            "ego_input_edit_strategy": None,
            "ego_correct_trajectory_strategy": "自车右转前减速，等支路车辆驶过后再完成转弯。",
        },
        {
            "candidate_id": "cand_2",
            "base_video_requirement": "丁字路口前视清晰，右侧支路有一辆可编辑车辆。",
            "retrieval_query": "白天丁字路口，自车右转，右侧支路清晰",
            "attack_edit_strategy": "调整右侧支路既有车辆的通过时机，使其与自车右转轨迹交叉。",
            "ego_input_edit_strategy": "把自车接近路口的输入轨迹改为稍晚到达。",
            "ego_correct_trajectory_strategy": "自车减速让行，确认支路车辆通过后再右转。",
        },
        {
            "candidate_id": "cand_3",
            "base_video_requirement": "丁字路口前视清晰，右侧支路入口有一辆等待车辆。",
            "retrieval_query": "白天丁字路口，自车右转，右侧支路清晰",
            "attack_edit_strategy": "让停放车辆起步左转进入主路，形成交叉冲突。",
            "ego_input_edit_strategy": None,
            "ego_correct_trajectory_strategy": "自车保持低速，待起步车辆离开冲突区后再右转。",
        },
    ],
}


def _result(_data):
    return {key: value for key, value in RESULT.items()}


def run(input_data: dict) -> dict:
    from .real import execute
    return execute(input_data, mock_result=_result)
