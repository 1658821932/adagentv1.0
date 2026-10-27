"""Typed contracts for scenario conception."""
from typing import Any
from pydantic import Field, field_validator
from tools.llm_runtime import NonEmptyStr, ToolModel, unwrap_result


class Input(ToolModel):
    scene_requirement: dict[str, Any] | NonEmptyStr = Field(
        description="Requirements from tool02, a successful envelope, or a direct natural-language requirement."
    )

    @field_validator("scene_requirement", mode="before")
    @classmethod
    def normalize_requirement(cls,value):
        value=unwrap_result(value)
        if not value:
            raise ValueError("A non-empty scene_requirement is required")
        return value


class EditIdea(ToolModel):
    operation: NonEmptyStr = Field(description="Proposed editing operation.")
    target_role: NonEmptyStr = Field(description="Participant role; do not invent an instance ID.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Proposed design parameters, not measurements.")


class Response(ToolModel):
    scene_plan: NonEmptyStr = Field(description="High-level scenario generation plan.")
    base_video_requirement: NonEmptyStr = Field(description="Description of source video that should be retrieved.")
    required_conditions: list[NonEmptyStr] = Field(default_factory=list, description="Requested road, weather and traffic conditions.")
    required_actors: list[NonEmptyStr] = Field(default_factory=list, description="Participant roles required in the source video.")
    edit_ideas: list[EditIdea] = Field(description="Proposed edits, with roles and parameters.")
    expected_effect: NonEmptyStr = Field(description="Intended scenario effect; actual policy response requires evaluation.")
    retrieval_query: NonEmptyStr = Field(description="Natural-language query for video_retriever.requirement.")


class Output(Response):
    source_requirement: dict[str, Any] | NonEmptyStr = Field(description="Original requirement attached by code.")
