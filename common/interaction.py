"""Two-vehicle conflict shape shared by the scenario planner and the edit planner.

Kept in one place so a candidate's interaction survives the handoff unchanged
instead of being re-derived from prose by a later tool.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from tools.llm_runtime import NonEmptyStr, ToolModel

APPROACH_FROM = ("left", "right", "opposite", "same_lane")
OTHER_MOTION = ("cross", "turn_in", "cut_in", "stop", "follow")
OTHER_ENTERS = ("ego_lane", "ego_path", "crossing")
EGO_MOTION = ("right_turn", "left_turn", "straight", "accelerate")
RELATIVE = ("head_on", "crossing", "same_direction")


class Interaction(ToolModel):
    """How the ego and one other vehicle meet."""

    other_approach_from: Literal[APPROACH_FROM] = Field(  # type: ignore[valid-type]
        description="Where the conflicting vehicle comes from, relative to the ego's travel direction.")
    other_motion: Literal[OTHER_MOTION] = Field(  # type: ignore[valid-type]
        description="What that vehicle does to create the conflict.")
    other_enters: Literal[OTHER_ENTERS] = Field(  # type: ignore[valid-type]
        description="Space the other vehicle moves into.")
    ego_motion: Literal[EGO_MOTION] = Field(  # type: ignore[valid-type]
        description="Ego's own recorded motion at the conflict.")
    relative: Literal[RELATIVE] = Field(  # type: ignore[valid-type]
        description="Relative heading of the two vehicles when they meet.")
    conflict_window_s: list[float] = Field(
        min_length=2, max_length=2,
        description="Seconds in which the conflict must happen, e.g. [2.0, 7.0].")

    @field_validator("conflict_window_s")
    @classmethod
    def order_window(cls, value):
        if len(value) != 2 or not all(isinstance(v, (int, float)) for v in value):
            raise ValueError("conflict_window_s needs two numbers")
        low, high = float(value[0]), float(value[1])
        if not 0.0 <= low < high:
            raise ValueError("conflict_window_s must be ordered and nonnegative")
        return [low, high]
