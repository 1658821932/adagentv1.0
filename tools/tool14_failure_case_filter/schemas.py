"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    scenario_pkl: NonEmptyStr = Field(validation_alias=AliasChoices("scenario_pkl","scenario_path"))
    result_dir: NonEmptyStr | None = None
    scene_token: NonEmptyStr | None = None
    timeout: PositiveInt = 120
    python: NonEmptyStr | None = None

class Output(OutputModel):
    is_failure_case: bool
    failure_cases: list[dict[str, Any]]
    collision: bool
    scene_count: PositiveInt = 1
    scene_results: list[dict[str, Any]] = Field(default_factory=list)
