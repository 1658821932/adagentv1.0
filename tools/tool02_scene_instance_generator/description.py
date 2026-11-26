"""Tool 02 purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name="scene_instance_generator",
    description="根据工具①的失败分析结果，生成统一目标场景与三个差异化实现候选（优先新增车辆方案），按候选声明资产需求并原样透传上游失败上下文与运动学档案。",
    when_to_use="将成功的 failure_analyzer 结果转化为可检索、可编辑的三个候选方案。",
    input_model=Input,
    output_model=Output,
    handoff={
        "input": "failure_analysis 接收成功的 failure_analyzer 结果、envelope 或绝对 result_ref。",
        "output": "选定一个候选，将其 retrieval_query 原样传给 video_retriever.requirement 并携带同一 candidate_id；检索后将完整 tool02 结果/ref 与 tool03 结果/ref 传给 edit_planner。",
    },
    rules=[
        "有目标车失败时优先采用新增车辆方案（cand_1 必须带 asset_requirement），需要新增车辆的候选声明 asset_requirement，编辑既有车辆的候选 asset_requirement 为 null。",
        "纯自车出界或无车碰撞时，所有候选的 attack_edit_strategy 和 asset_requirement 必须为 null。",
        "failure_context 与 kinematics 由程序从工具①原样透传，不被大模型改写。",
        "retrieval_query 为 6–48 字精简检索词，只含环境、最小道路形态、自车方向与所需空位，禁用编辑/冲突/他车动作词。",
    ],
)
