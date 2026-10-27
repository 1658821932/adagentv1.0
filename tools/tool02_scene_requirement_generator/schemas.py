"""Executable input, model-response and final-output contracts."""
from typing import Any
from pydantic import Field, field_validator
from tools.llm_runtime import NonEmptyStr, ToolModel, unwrap_result


class Input(ToolModel):
    failure_analysis: dict[str, Any] = Field(
        description="Non-empty failure analysis; accepts tool01 result or its successful envelope."
    )

    @field_validator("failure_analysis", mode="before")
    @classmethod
    def normalize_analysis(cls, value):
        value = unwrap_result(value, "failure_analysis")
        if not isinstance(value, dict) or not value:
            raise ValueError("A non-empty failure_analysis object is required")
        return value


class Response(ToolModel):
    scene_requirement: NonEmptyStr = Field(description="Target scenario to generate; a design goal, not an observed result.")
    failure_type: str = Field(default="", description="Failure category supported by the source analysis.")
    failure_reason: str = Field(default="", description="Source-supported failure reason, preserving uncertainty.")
    required_conditions: list[NonEmptyStr] = Field(description="Desired road, weather and traffic conditions.")
    required_actors: list[NonEmptyStr] = Field(description="Required participant roles.")
    risk_factors: list[NonEmptyStr] = Field(description="Risk factors supported by the analysis.")
    target_effect: NonEmptyStr = Field(description="Driving-policy behavior the generated scenario should test.")
    hazard_summary: str = Field(default="", description="Concise hazard description.")


class Output(Response):
    source_analysis: dict[str, Any] = Field(description="Original analysis attached by code, never reconstructed by the LLM.")
