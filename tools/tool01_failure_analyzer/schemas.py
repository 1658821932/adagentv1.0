"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    token: NonEmptyStr = Field(description='WorldEngine scene token to analyze (short or full identifier).')
    simulation: NonEmptyStr | None = Field(
        default=None,
        description="Which closed-loop simulation record produced the official metrics: "
                    "'<experiment>/<batch>', a unique batch name, or an absolute archive directory "
                    "containing scene_index.csv. A scene may be simulated more than once with "
                    "different policies, so this selects the run while token selects the scene.")
    timeout: PositiveInt = Field(default=300, description="Analysis process timeout in seconds.")

class Output(OutputModel):
    failure_analysis: dict[str, Any] = Field(description='Compact single-scene diagnosis: failure_type, failure_reason, key_scene_factors (up to 5), official_metrics and uncertainty.')
    token: str = Field(default='', description='Full WorldEngine scene token that was analyzed.')
    simulation: str | None = Field(default=None, description='Closed-loop simulation record that produced the official metrics.')
    evidence_dir: str = Field(default="", description="Directory holding the evidence used for the analysis (images and numeric tables), for optional inspection.")
