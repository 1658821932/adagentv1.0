"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    image_dir: NonEmptyStr
    output_path: NonEmptyStr
    backend: Literal["dggt_yolo"] = "dggt_yolo"
    scene_name: NonEmptyStr = "000"
    sequence_length: Literal[40] = 40
    start_idx: Annotated[int, Field(ge=0)] = 0
    weights: NonEmptyStr = "yolo11m.pt"
    conf: Annotated[float, Field(ge=0, le=1)] = 0.25
    timeout: PositiveInt = 3600
    vis_dir: NonEmptyStr | None = None

class Output(OutputModel):
    instances: list[dict[str, Any]]
    instance_info_path: NonEmptyStr
