"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    token: NonEmptyStr = Field(validation_alias=AliasChoices("token","scene_token"))
    config_path: NonEmptyStr | None = None
    run_name: NonEmptyStr | None = None
    resume: bool = True
    timeout: PositiveInt = 1800

class Output(OutputModel):
    failure_analysis: dict[str, Any]
