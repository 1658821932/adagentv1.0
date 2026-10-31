"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    rendered_video: NonEmptyStr
    scene_requirement: NonEmptyStr | dict[str, Any]
    frame_count: Annotated[int, Field(ge=8)] = 8
    frame_paths: list[NonEmptyStr] | None = None
    edited_scene: dict[str, Any] | None = None
    visual_evaluation: dict[str, Any] | None = None
    local_evaluator: NonEmptyStr | list[NonEmptyStr] | None = None
    scene_config: NonEmptyStr | None = None
    scene_ply: NonEmptyStr | None = None
    ignored_reconstruction_artifacts: list[NonEmptyStr] = Field(default_factory=list)
    mock_force_fail: bool = False

    @model_validator(mode="before")
    @classmethod
    def upstream(cls, value):
        if not isinstance(value,dict):
            return value
        data=dict(value)
        if "requirement" in data:
            data.setdefault("scene_requirement", data.pop("requirement"))
        scene=data.get("edited_scene") or {}
        if isinstance(scene,dict):
            scene=scene.get("result",scene)
            if isinstance(scene,dict) and "rendered_video" in scene:
                data.setdefault("rendered_video",scene["rendered_video"])
        return data

class Output(OutputModel):
    passed: bool
    requirement_consistency: bool
    geometry_valid: bool
    physics_valid: bool
    traffic_logic_valid: bool
    visual_valid: bool
    issues: list[str]
