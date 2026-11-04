"""Deterministic merged scene-design and retrieval-strategy fixture."""
from .real import execute


def _result(data):
    analysis, hints = data.failure_analysis, data.variation_hints
    reason = analysis.get("failure_reason") or analysis["wrong_behavior"]
    road = hints.get("road_layout") or "与目标交互匹配、行驶区域清晰可见的道路场景"
    actor = hints.get("target_actor") or hints.get("actor_role") or "一辆他车"
    context = "；".join([road] + [hints[k] for k in ("lighting", "weather", "traffic_density") if hints.get(k)])
    return {
        "scene_requirement": f"构造与以下失败模式对应的新交互：{reason}。自车接近交互区域，{actor}的运动与自车路线产生冲突。场景背景：{context}。",
        "target_effect": analysis.get("expected_behavior") or "自车根据交互及时减速、让行或停车，安全通过。",
        "candidates": [
            {"base_video_requirement": context + "，有一辆可用于构造交互的行驶车辆。", "edit_strategy": "调整已有这一辆他车的运动轨迹和经过时机，按需要配合自车轨迹调整。"},
            {"base_video_requirement": context + "，目标交互区域空闲，没有相关车辆也可。", "edit_strategy": "增加一辆他车并设置其运动轨迹，形成目标交互；按需要调整自车接近轨迹。"},
            {"base_video_requirement": context + "，路侧或侧向入口有一辆停放、等待的车辆。", "edit_strategy": "修改已有这一辆车辆的状态或位置，并调整起步和经过轨迹以形成交互。"},
        ],
    }


def run(input_data: dict) -> dict:
    return execute(input_data, mock_result=_result)
