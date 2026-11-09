"""Native WE replay requires a prepared, version-correct scenario package."""
from typing import Annotated, Literal
from pydantic import Field
from common.models import InputModel, OutputModel, NonEmptyStr
from mcp_servers.scene_editing.adapter.schemas import SimulationInput

class Input(InputModel):
    simulation_input: SimulationInput = Field(description="Prepared WE native scenario/asset paths for this exact scene version. Frozen frames.json or an expert CSV alone is not a native package.")
    scene_id: NonEmptyStr = Field(description="Exact key of ONE scene in the native scenario pickle.")
    cameras: list[NonEmptyStr] | None = Field(default=None,min_length=1,description="Native camera names to replay; null uses all scene cameras.")
    timeout: Annotated[int,Field(gt=0)] = 1800

class Output(OutputModel):
    mode: Literal['real','mock']
    scene_id: NonEmptyStr
    result_dir: NonEmptyStr
    videos: list[NonEmptyStr]
    replay_report: NonEmptyStr
    checks: dict
