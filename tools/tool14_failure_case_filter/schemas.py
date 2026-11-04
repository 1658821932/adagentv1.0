"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated
from pydantic import Field, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]

class Input(InputModel):
    scenario_pkl: NonEmptyStr = Field(validation_alias=AliasChoices("scenario_pkl","scenario_path"))
    result_dir: NonEmptyStr = Field(description="policy_runner result directory containing official PDM CSV and execution metadata.")
    scene_token: NonEmptyStr | None = None
    timeout: PositiveInt = 120
    python: NonEmptyStr | None = None

class Output(OutputModel):
    is_failure_case: bool
    failure_cases: list[dict[str, Any]]
    pdm_scores_file: NonEmptyStr
    scene_count: PositiveInt = 1
    scene_results: list[dict[str, Any]] = Field(default_factory=list)
