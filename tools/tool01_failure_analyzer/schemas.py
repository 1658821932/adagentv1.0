"""Executable Agent-tool input and output contracts for Tool 01."""
from typing import Any, Annotated
from pydantic import Field
from tools.contracts import InputModel, OutputModel, NonEmptyStr

PositiveInt = Annotated[int, Field(gt=0)]


class Input(InputModel):
    token: NonEmptyStr = Field(description="WorldEngine scene token to analyze (short or full identifier).")
    simulation: NonEmptyStr | None = Field(
        default=None,
        description="Which closed-loop simulation record produced the official metrics: "
                    "'<experiment>/<batch>', a unique batch name, or an absolute archive directory "
                    "containing scene_index.csv. Required for real analysis; mock may omit it.",
    )
    timeout: PositiveInt = Field(
        default=300,
        description="Total analysis budget in seconds covering data extraction, BEV rendering, and model calls.",
    )


class Output(OutputModel):
    token: str = Field(description="Full WorldEngine scene token that was analyzed.")
    violation_step: int = Field(description="Official first violation frame.")
    simulation: str | None = Field(default=None, description="Closed-loop simulation record.")
    map_location: str | None = Field(default=None, description="nuPlan map name for this scene.")
    evidence_dir: str = Field(default="", description="Directory holding evidence artifacts for this analysis.")
    bev_image: str = Field(default="", description="Path to the rendered BEV overview PNG.")
    failure: dict[str, Any] = Field(
        description="Failure verdict and script: outcome (collision / road_departure), "
                    "failure_driver (collision_with_vehicle / collision_non_vehicle / "
                    "departure_by_other / departure_ego_only), target_track, target_class, "
                    "map_context, ego_behavior, other_behavior, and summary.",
    )
    kinematics: dict[str, Any] = Field(
        description="Deterministic physical measurements: ego, other (or null), and trajectory_ref.",
    )
    environment: dict[str, Any] = Field(
        description="Environment: time (scene timestamp), weather, and extra static context.",
    )
    verification: dict[str, Any] = Field(
        description="Visual verification record: sample_frames and notes.",
    )
