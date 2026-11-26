"""Tool 01 purpose and invocation guidance; field contracts come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name="failure_analyzer",
    description="分析单个闭环仿真失败场景：程序提取官方指标、物理运动学与高清 BEV，BEV VLM 判定最小道路形态与唯一目标车剧本，8 相机可选补充环境。",
    when_to_use="分析已有 WorldEngine 闭环仿真中的单个失败场景。",
    input_model=Input,
    output_model=Output,
    handoff={
        "input": "同时提供 simulation 和 token：前者选择仿真批次，后者选择其中场景。",
        "output": "返回 violation_step、bev_image、failure（含 outcome、failure_driver、target_track、target_class、map_context 及剧本）、kinematics（自车/目标车物理量与轨迹文件）、environment 和 verification。",
    },
    rules=[
        "token 使用用户提供或 policy_runner→failure_case_filter 获得的仿真场景 ID，支持短 ID 或完整 ID；不使用视频检索 scene_ref。",
        "同一场景可能有多次仿真，必须指定 simulation；使用 '<experiment>/<batch>'、唯一批次名或归档绝对路径。",
        "调用期间不要重复启动；官方指标与物理测量不可被视觉模型改写。",
    ],
)
