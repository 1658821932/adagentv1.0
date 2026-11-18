"""Executable Agent-tool input and output contracts."""
from typing import Annotated, Literal
from pydantic import Field, model_validator
from tools.contracts import InputModel, OutputModel, NonEmptyStr
from common.scene_refs import SceneReference
from mcp_servers.sentry_video_search.adapter.config import DEFAULT_RETRIEVAL_TIMEOUT_S
PositiveInt = Annotated[int, Field(gt=0)]

class Input(InputModel):
    requirement: NonEmptyStr = Field(description='Natural-language source-video search description.')
    library: Literal['default'] = Field(default='default', description='Local index selection; only the default library is available.')
    timeout: PositiveInt = Field(default=DEFAULT_RETRIEVAL_TIMEOUT_S, description='Retrieval process timeout in seconds.')
    mock_video_path: NonEmptyStr | None = Field(default=None, description='Mock-only video path; it does not provide a real indexed scene.')
    candidate_id: NonEmptyStr | None = Field(default=None, description='tool02 candidate this requirement came from; echoed back so the plan can be traced to its variant.')

class Output(OutputModel):
    video_path: NonEmptyStr = Field(description='Exported search clip path.')
    match_score: float = Field(allow_inf_nan=False, description='Retrieval similarity score, not scenario-validation confidence.')
    mode: NonEmptyStr = Field(description='Retrieval execution mode.')
    source_file: NonEmptyStr | None = Field(default=None, description='Original indexed camera video.')
    source_segment: dict[str, float] | None = Field(default=None, description='Index-match start/end in source-video seconds; not exported clip bounds.')
    scene_ref: SceneReference | None = Field(default=None, description='Full source-scene identity, camera/image availability and annotation/Gaussian paths. Required by real retrieval; mock may return null. manifest_path locates ordered images, timestamps and calibrations.')
    candidate_id: NonEmptyStr | None = Field(default=None, description='The tool02 candidate id this retrieval was launched for, echoed unchanged.')

    @model_validator(mode='after')
    def real_result_requires_binding(self):
        if self.mode != 'mock' and (not self.source_file or self.scene_ref is None):
            raise ValueError('Real retrieval requires source_file and a bound scene_ref')
        return self
