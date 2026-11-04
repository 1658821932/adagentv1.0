"""Detailed new scenario requirements inspired by one saved failure category."""
from typing import Any
from pydantic import AliasChoices, Field, field_validator
from tools.llm_runtime import NonEmptyStr, ToolModel, unwrap_result


def normalized(text):
    return " ".join(text.casefold().split())


class Input(ToolModel):
    failure_analysis: dict[str, Any] = Field(description="Saved failure evidence or a successful tool01 envelope. Accepts failure_reason or a saved group with wrong_behavior and id. Keep original case IDs.")
    variation_hints: dict[str, NonEmptyStr] = Field(default_factory=dict, validation_alias=AliasChoices("variation_hints", "parameter_assignment"), description="Optional qualitative variation directions, e.g. traffic_density: sparse, target_actor: crossing bus. Legacy parameter_assignment is accepted. No distance/speed/timing values are required.")
    existing_instances: list[NonEmptyStr] = Field(default_factory=list, description="Previous requirement summaries or successful tool02 outputs. Exact normalized descriptions are rejected; semantic diversity remains a model suggestion.")

    @field_validator("failure_analysis", mode="before")
    @classmethod
    def normalize_analysis(cls, value):
        value = unwrap_result(value, "failure_analysis")
        if not isinstance(value, dict) or not isinstance(value.get("failure_reason") or value.get("wrong_behavior"), str) or not (value.get("failure_reason") or value.get("wrong_behavior")).strip():
            raise ValueError("failure_analysis requires failure_reason or group wrong_behavior")
        category = value.get("group_id") or value.get("id") or value.get("failure_type")
        if category is not None and not (isinstance(category, str) and category.strip() or isinstance(category, list) and category and all(isinstance(x,str) and x.strip() for x in category)):
            raise ValueError("group_id / failure_type must be non-empty text or a list of non-empty text")
        if "supporting_case_ids" in value and (not isinstance(value["supporting_case_ids"], list) or not all(isinstance(item, str) and item.strip() for item in value["supporting_case_ids"])):
            raise ValueError("supporting_case_ids must be a list of non-empty source IDs")
        if "cases" in value and (not isinstance(value["cases"], list) or not all(isinstance(item, dict) for item in value["cases"])):
            raise ValueError("cases must be a list of source case objects")
        # Keep legacy upstream outputs usable without forwarding this field.
        value = {key: item for key, item in value.items() if key != "uncertainty"}
        if isinstance(value.get("forensics"), dict):
            value["forensics"] = {key: item for key, item in value["forensics"].items() if key != "uncertainty"}
        return value

    @field_validator("existing_instances", mode="before")
    @classmethod
    def normalize_previous(cls, value):
        if not isinstance(value, list):
            return value
        return [(unwrap_result(item).get("scene_requirement") or unwrap_result(item).get("scene_instance", "")) if isinstance(item, dict) else item for item in value]


class Candidate(ToolModel):
    base_video_requirement: NonEmptyStr = Field(description="Natural-language description of the BASE video to retrieve. Pass this string verbatim to video_retriever.requirement. Describe visible road context and actors needed for THIS strategy; an empty road/intersection is valid when adding a vehicle. Do not require the final hazard to already exist.")
    edit_strategy: NonEmptyStr = Field(description="Qualitative strategy for this base video: add, delete or modify a vehicle and/or its trajectory; optionally adjust ego trajectory. Prefer changing at most one non-ego vehicle. No fixed object IDs or numeric trajectories are needed.")


class Response(ToolModel):
    scene_requirement: NonEmptyStr = Field(description="ONE target scene with context, participant interaction and event progression. All three candidates construct this same target.")
    target_effect: NonEmptyStr = Field(description="What ego should do in the target scene.")
    candidates: list[Candidate] = Field(min_length=3, max_length=3, description="Exactly three alternative base-video retrieval descriptions paired with corresponding editing strategies.")


class Output(Response):
    pass
