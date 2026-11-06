"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class VisualEvaluation(OutputModel):
    """Untrusted evaluator output: no string/number-to-boolean coercion."""
    requirement_consistency: bool
    visual_valid: bool
    geometry_valid: bool
    physics_valid: bool
    traffic_logic_valid: bool
    confidence: Annotated[float, Field(ge=0, le=1)] = 0.0
    issues: list[str] = Field(default_factory=list)
    evidence: list[Any] = Field(default_factory=list)
    ignored_reconstruction_artifacts: list[str] = Field(default_factory=list)
    model: str | None = None
    sampled_frames: Annotated[int, Field(ge=0)] = 0


class Input(InputModel):
    rendered_video: NonEmptyStr = Field(description='Actual edited-render video to inspect.')
    scene_requirement: NonEmptyStr | dict[str, Any] = Field(description='Original target-scene requirement.')
    frame_count: Annotated[int, Field(ge=8)] = Field(default=8, description='Uniformly sampled frames for VLM review; at least eight.')
    frame_paths: list[NonEmptyStr] | None = Field(default=None, description='Optional already extracted keyframes of this same edited render.')
    edited_scene: dict[str, Any] | None = Field(default=None, description='Optional editing-result metadata; does not replace the rendered video.')
    visual_evaluation: VisualEvaluation | None = Field(default=None, description='Optional result from an actual external visual evaluator; otherwise use the configured VLM.')
    local_evaluator: NonEmptyStr | list[NonEmptyStr] | None = Field(default=None, description='Optional local evaluator command; receives scene/video context and returns JSON.')
    scene_config: NonEmptyStr | None = Field(default=None, description='Optional scene configuration file to check for existence and valid JSON.')
    scene_ply: NonEmptyStr | None = Field(default=None, description='Optional scene PLY artifact to check for existence.')
    ignored_reconstruction_artifacts: list[NonEmptyStr] = Field(default_factory=list, description='Known source-render defects excluded from edit-quality issues unless they hide the target.')
    mock_force_fail: bool = Field(default=False, description='Mock-only forced validation failure.')

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
    passed: bool = Field(description='True only when all five checks pass and issues is empty.')
    requirement_consistency: bool = Field(description='Rendered scene matches the requested scenario.')
    geometry_valid: bool = Field(description='Visible vehicle placement, scale and road contact are plausible.')
    physics_valid: bool = Field(description='Visible motion is plausible; not a dynamics guarantee.')
    traffic_logic_valid: bool = Field(description='Visible participant interaction is plausible.')
    visual_valid: bool = Field(description='Target is recognizable and visually continuous without edit-induced damage.')
    issues: list[str] = Field(description='Actionable failed checks and their evidence.')
