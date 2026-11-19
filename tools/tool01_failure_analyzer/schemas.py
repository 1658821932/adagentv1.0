"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated
from pydantic import Field
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]

class Input(InputModel):
    token: NonEmptyStr = Field(description='WorldEngine scene token to analyze (short or full identifier).')
    simulation: NonEmptyStr | None = Field(
        default=None,
        description="Which closed-loop simulation record produced the official metrics: "
                    "'<experiment>/<batch>', a unique batch name, or an absolute archive directory "
                    "containing scene_index.csv. A scene may be simulated more than once with "
                    "different policies, so this selects the run while token selects the scene. "
                    "Required for real analysis; mock may omit it.")
    timeout: PositiveInt = Field(
        default=300,
        description="总分析预算（秒）。数据子进程、图像准备、以及每一次模型请求（含重试）共用这一份预算，"
                    "超时后整组进程被终止。")

class Output(OutputModel):
    token: str = Field(description='Full WorldEngine scene token that was analyzed.')
    violation_step: int = Field(description='Official first violation frame used for behavior analysis.')
    simulation: str | None = Field(default=None, description='Closed-loop simulation record that produced the official metrics.')
    map_location: str | None = Field(default=None, description='nuPlan map name for this scene; used when a map lookup is needed.')
    evidence_dir: str = Field(default="", description='Directory holding the evidence used for the analysis, for optional inspection.')
    failure: dict[str, Any] = Field(description='Final failure facts: outcome, collision_confirmed, interaction (rear_end / opposite_direction / cut_in / crossing / side_swipe / near_miss / single_vehicle) and ego_cause. near_miss means a mapped nearby vehicle exists without an official collision.')
    attack: dict[str, Any] = Field(description='Final two-vehicle structure: map_context, ego_action, ego_lane_keeping, other_origin, other_action and other_lane_keeping.')
    environment: dict[str, Any] = Field(description='Environment: time (from the scene timestamp), weather (clear / cloudy / rainy / foggy / snowy / unknown) and extra (one sentence of visual context).')
    verification: dict[str, Any] = Field(description='Three-frame eight-camera verification: sample_frames, corrections from measured to final values, and notes.')
