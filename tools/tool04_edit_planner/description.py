"""Tool 04 purpose and invocation guidance; field contracts come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name="edit_planner",
    description="双 BEV 对照模仿规划器：将工具①的原失败仿真 BEV 与工具③新检索真实底场景的标准 16 帧 BEV 对照，结合 3DGS 原采集轨迹约束审查可编辑性，并输出可直接交给工具⑤的确定性 attack_input。",
    when_to_use="在 video_retriever 绑定真实底场景后、进入 attack_vehicle_editor 之前，验证底场景是否能复刻原危险场景并生成针对该底场景的模仿规划。",
    input_model=Input,
    output_model=Output,
    handoff={
        "input": "传入 scenario_result（工具②结果/ref）、selected candidate_id 和 retrieval_result（工具③结果/ref）。",
        "output": "suitability=ready 时（无论双车 add_vehicle/edit_existing 还是单车 ego_only），均直接将返回的 attack_input 原样传给 attack_vehicle_editor；若 unsuitable，使用 retrieval_suggestion 重新检索（并将该无效底场景加入 exclude_scene_tokens）。",
    },
    rules=[
        "3DGS 高斯点云围绕真实底场景采集轨迹重建，图 2 优先叠加当前策略闭环仿真轨迹（无缓存时回退原采集轨迹）；自车不可改道去未采集支路，只能沿原路径微调速度或小幅横向偏移。",
        "候选带 asset_requirement 时锁定 operation=add_vehicle，由工具⑤按原生尺寸和官方道路校验选择资产；仅在用户明确指定时传入车型约束；编辑既有车时锁定 operation=edit_existing；单车失败时锁定 operation=ego_only 并按基线 PDM 自动决定 prepare_ego_history。",
        "ready 时双车默认 prepare_ego_history=false，且 attack_input.constraints 仅施加 no_ego_contact_until_s=3.0（target_window_s 置为 null，不锁死工具⑤求解时间窗）。",
    ],
)
