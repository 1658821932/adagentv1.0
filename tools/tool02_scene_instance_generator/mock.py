"""Deterministic mock fixture for Tool 02."""

RESULT = {
    "target_scene": "晴朗白天，弯道段，自车低速行驶末段近直行，对向一辆大型车左转迎面驶来。",
    "candidates": [
        {
            "candidate_id": "cand_1",
            "base_video_requirement": "晴朗白天弯道，对向车道空旷清晰，适合新增大型车。",
            "retrieval_query": "晴朗白天弯道，自车行驶，对向车道清晰",
            "asset_requirement": {"vehicle_class": "bus"},
            "attack_edit_strategy": "在对向车道新增一辆大型车，使其高速左转进入弯道与自车末段直行交汇。",
            "ego_input_edit_strategy": None,
        },
        {
            "candidate_id": "cand_2",
            "base_video_requirement": "白天双向弯道，对向弯道入口空旷可新增车辆。",
            "retrieval_query": "白天双向弯道，对向入口空旷清晰",
            "asset_requirement": {"vehicle_class": "bus"},
            "attack_edit_strategy": "在对向弯道入口新增大型车并设定左转时机，使其在弯道中段逼近自车。",
            "ego_input_edit_strategy": "微调自车前1.5秒进入弯道的速度以同步到达冲突点。",
        },
        {
            "candidate_id": "cand_3",
            "base_video_requirement": "晴朗白天弯道，对向车道已有一辆可编辑的大型车。",
            "retrieval_query": "晴朗白天弯道，对向车道视野清晰",
            "asset_requirement": None,
            "attack_edit_strategy": "调整对向既有大型车的左转轨迹与速度，使其在自车出弯段迎面交汇。",
            "ego_input_edit_strategy": "调整自车入口速度与对向既有大型车同步交汇。",
        },
    ],
}


def _result(data):
    driver = data.failure_analysis["failure"]["failure_driver"]
    if driver in ("departure_ego_only", "collision_non_vehicle"):
        return {
            "target_scene": "晴朗白天，弯道段，自车减速行驶并保持在弯道内。",
            "candidates": [
                {
                    "candidate_id": "cand_1",
                    "base_video_requirement": "晴朗白天弯道，自车车道视野清晰无遮挡。",
                    "retrieval_query": "晴朗白天弯道，前方车道清晰",
                    "asset_requirement": None,
                    "attack_edit_strategy": None,
                    "ego_input_edit_strategy": None,
                },
                {
                    "candidate_id": "cand_2",
                    "base_video_requirement": "白天双向弯道，弯道弧度明显且路面清晰。",
                    "retrieval_query": "白天明显弯道，路面清晰",
                    "asset_requirement": None,
                    "attack_edit_strategy": None,
                    "ego_input_edit_strategy": "适当提高自车入弯前速度以复现弯道外偏压力。",
                },
                {
                    "candidate_id": "cand_3",
                    "base_video_requirement": "白天连续弯道，道路边界清晰。",
                    "retrieval_query": "白天连续弯道，道路边界清晰",
                    "asset_requirement": None,
                    "attack_edit_strategy": None,
                    "ego_input_edit_strategy": "微调自车前1.5秒航向略偏外侧以复现压边风险。",
                },
            ],
        }
    return {key: value for key, value in RESULT.items()}


def run(input_data: dict) -> dict:
    from .real import execute
    return execute(input_data, mock_result=_result)
