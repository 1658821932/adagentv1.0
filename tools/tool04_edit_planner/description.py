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
        "output": "suitability=ready 且为有车失败时，直接将返回的 attack_input 原样传给 attack_vehicle_editor；若 unsuitable，使用 retrieval_suggestion 重新检索；若为单车失败且 ready，attack_input 为 null，不调用工具⑤。",
    },
    rules=[
        "3DGS 高斯点云围绕真实底场景原采集轨迹 E0..E15 重建，自车不可改道去未采集支路，只能微调速度或小幅横向偏移。",
        "候选带 asset_requirement 时锁定 operation=add_vehicle，由工具⑤按原生尺寸和官方道路校验选择资产；仅在用户明确指定时传入车型约束；编辑既有车时锁定 operation=edit_existing。",
        "ready 时自动输出落在 [1.5, 7.5]s 内的 target_window_s 并填入 attack_input.constraints。",
    ],
)
