"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    token: NonEmptyStr = Field(description='WorldEngine scene token to analyze.')
    config_path: NonEmptyStr | None = Field(default=None, description='Optional WFA configuration for archived evaluation data and scene metadata.')
    run_name: NonEmptyStr | None = Field(default=None, description='Optional analysis run label.')
    resume: bool = Field(default=True, description='Reuse an existing completed analysis when available.')
    timeout: PositiveInt = Field(default=1800, description='Analysis process timeout in seconds.')

class Output(OutputModel):
    failure_analysis: dict[str, Any] = Field(description='Single-scene failure_type, failure_reason and scene factors; forensics retains observations, candidate assessments and full evidence/report paths.')
