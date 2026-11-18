"""Executable Agent-tool input and output contracts."""
from typing import Annotated, Literal
from pydantic import Field, field_validator, model_validator
from tools.contracts import InputModel, OutputModel, NonEmptyStr
from common.scene_refs import SceneReference
from mcp_servers.sentry_video_search.adapter.config import DEFAULT_RETRIEVAL_TIMEOUT_S
PositiveInt = Annotated[int, Field(gt=0)]

class Input(InputModel):
    requirement: NonEmptyStr = Field(max_length=60, description='Compact source-video search query from tool02 candidate.retrieval_query; weather/time when material, road form, ego direction and required clear space only.')
    library: Literal['default'] = Field(default='default', description='Local index selection; only the default library is available.')
    timeout: PositiveInt = Field(default=DEFAULT_RETRIEVAL_TIMEOUT_S, description='Retrieval process timeout in seconds.')
    mock_video_path: NonEmptyStr | None = Field(default=None, description='Mock-only video path; it does not provide a real indexed scene.')
    candidate_id: NonEmptyStr | None = Field(default=None, description='tool02 candidate this requirement came from; echoed back so the plan can be traced to its variant.')

    @field_validator('requirement')
    @classmethod
    def requirement_is_compact(cls, value):
        if len(value) > 60:
            raise ValueError('requirement must be a compact retrieval_query (at most 60 characters), not a full scenario description')
        return value


class AlternativeScene(OutputModel):
    rank: Annotated[int, Field(ge=2, le=3)] = Field(description='Rerank position after the winner.')
    video_path: NonEmptyStr = Field(description='Exported clip for this ranked alternative.')
    match_score: float = Field(allow_inf_nan=False, description='Rerank similarity score for this alternative.')
    source_file: NonEmptyStr = Field(description='Original indexed camera video.')
    source_segment: dict[str, float] | None = Field(default=None, description='Index-match start/end in source-video seconds.')
    scene_ref: SceneReference = Field(description='Bound source scene for this alternative.')

class Output(OutputModel):
    video_path: NonEmptyStr = Field(description='Exported search clip path.')
    match_score: float = Field(allow_inf_nan=False, description='Retrieval similarity score, not scenario-validation confidence.')
    mode: NonEmptyStr = Field(description='Retrieval execution mode.')
    source_file: NonEmptyStr | None = Field(default=None, description='Original indexed camera video.')
    source_segment: dict[str, float] | None = Field(default=None, description='Index-match start/end in source-video seconds; not exported clip bounds.')
    scene_ref: SceneReference | None = Field(default=None, description='Full source-scene identity, camera/image availability and annotation/Gaussian paths. Required by real retrieval; mock may return null. manifest_path locates ordered images, timestamps and calibrations.')
    candidate_id: NonEmptyStr | None = Field(default=None, description='The tool02 candidate id this retrieval was launched for, echoed unchanged.')
    alternatives: list[AlternativeScene] = Field(default_factory=list, max_length=2, description='Distinct ranked scenes 2 and 3, already bound and available for tool04 fallback.')

    @model_validator(mode='after')
    def real_result_requires_binding(self):
        if self.mode != 'mock' and (not self.source_file or self.scene_ref is None):
            raise ValueError('Real retrieval requires source_file and a bound scene_ref')
        if self.mode != 'mock' and any(item.scene_ref is None for item in self.alternatives):
            raise ValueError('Every retrieved alternative requires a bound scene_ref')
        return self
