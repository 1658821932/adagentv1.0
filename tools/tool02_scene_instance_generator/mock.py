"""Deterministic fixture for the compact scene-requirement contract.

It is a fixed legal example of the output shape; it does not derive or claim a new
generation mechanism.
"""

RESULT = {
    "target_scene": "多云白天，丁字路口，自车右转进入主路，右侧支路一辆车左转驶入。",
    "attack": {
        "map_context": "t_junction",
        "ego_action": "turn_right",
        "other_origin": "right_arm",
        "other_action": "turn_left",
    },
    "candidates": [
        {
            "candidate_id": "cand_1",
            "base_video_requirement": "丁字路口前视清晰，右侧支路有一辆正常行驶、可改轨迹的车辆。",
            "attack_edit_strategy": "调整右侧支路车辆的通过时机，使其与自车右转轨迹交叉。",
            "ego_input_edit_strategy": None,
            "ego_correct_trajectory_strategy": "自车右转前减速，等支路车辆驶过后再完成转弯。",
        },
        {
            "candidate_id": "cand_2",
            "base_video_requirement": "丁字路口前视清晰，右侧支路空闲、允许新增一辆车辆。",
            "attack_edit_strategy": "在右侧支路新增一辆左转车辆并设定其行驶轨迹。",
            "ego_input_edit_strategy": "把自车接近路口的输入轨迹改为稍晚到达。",
            "ego_correct_trajectory_strategy": "自车减速让行，确认支路车辆通过后再右转。",
        },
        {
            "candidate_id": "cand_3",
            "base_video_requirement": "丁字路口前视清晰，右侧支路入口有一辆停放或等待的车辆。",
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
