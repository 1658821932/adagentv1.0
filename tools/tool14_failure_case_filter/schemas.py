"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated
from pydantic import Field, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]

class Input(InputModel):
    scenario_pkl: NonEmptyStr = Field(validation_alias=AliasChoices('scenario_pkl', 'scenario_path'), description='Scenario pickle whose identities correspond to the evaluated run.')
    result_dir: NonEmptyStr = Field(description='policy_runner output containing official PDM CSV and execution metadata.')
    scene_token: NonEmptyStr | None = Field(default=None, description='Optional full scene identity or unambiguous 16-character suffix.')
    timeout: PositiveInt = Field(default=120, description='Score-reading process timeout in seconds.')
    python: NonEmptyStr | None = Field(default=None, description='Optional interpreter for reading native scenario metadata.')

class Output(OutputModel):
    is_failure_case: bool = Field(description='Whether any selected scene has official PDM total score zero.')
    failure_cases: list[dict[str, Any]] = Field(description='Zero-PDM scene identities, scenario paths and scores.')
    pdm_scores_file: NonEmptyStr = Field(description='Official PDM CSV actually read.')
    scene_count: PositiveInt = Field(default=1, description='Number of matched evaluated scenes.')
    scene_results: list[dict[str, Any]] = Field(default_factory=list, description='Per-scene official score and failure selection.')
