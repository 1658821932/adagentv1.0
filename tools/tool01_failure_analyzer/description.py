"""Tool 01 purpose and invocation guidance; field contracts come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name="failure_analyzer",
    description="分析一个闭环仿真失败场景，用官方指标、轨迹和地图确定失败与交互结构，并用违规帧画面补充环境。",
    when_to_use="分析已有WorldEngine仿真中的单个失败场景。",
    input_model=Input,
    output_model=Output,
    handoff={
        "input": "同时提供simulation和token：前者选择仿真批次，后者选择其中场景。",
        "output": "返回violation_step、failure、attack、environment和verification；attack包含自车/他车动作及车道保持，verification保存三帧核对和修正记录。",
    },
    rules=[
        "token使用用户提供或policy_runner→failure_case_filter获得的仿真场景ID，支持短ID或完整ID；不使用视频检索scene_ref。",
        "同一场景可能有多次仿真，必须指定simulation；使用'<experiment>/<batch>'、唯一批次名或归档绝对路径。",
        "调用期间不要重复启动；官方指标、违规帧、轨迹、地图和交互分类保持不变，视觉模型只补充weather、extra和notes。",
    ],
)
