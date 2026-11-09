"""Detailed new scenario requirements inspired by one saved failure category."""
from typing import Any, Optional
from pydantic import Field, field_validator
from tools.llm_runtime import NonEmptyStr, ToolModel, unwrap_result


def normalized(text):
    return " ".join(text.casefold().split())


class Input(ToolModel):
    failure_analysis: dict[str, Any] = Field(description="Saved failure category or tool01 result with nonempty failure_reason or wrong_behavior. Preserve supplied case IDs.")
    variation_hints: dict[str, NonEmptyStr] = Field(default_factory=dict, description="Optional qualitative variation directions, e.g. traffic_density: sparse, target_actor: crossing bus. No distance/speed/timing values are required.")
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
        # Omit obsolete analysis metadata.
        value = {key: item for key, item in value.items() if key != "uncertainty"}
        if isinstance(value.get("forensics"), dict):
            value["forensics"] = {key: item for key, item in value["forensics"].items() if key != "uncertainty"}
        return value

    @field_validator("existing_instances", mode="before")
    @classmethod
    def normalize_previous(cls, value):
        if not isinstance(value, list):
            return value
        return [(unwrap_result(item).get("scene_requirement") or "") if isinstance(item, dict) else item for item in value]


class Candidate(ToolModel):
    base_video_requirement: NonEmptyStr = Field(description="Source-video scene description for retrieval. Describe ONLY observable road layout and other vehicles present. Do NOT include ego behavior or editing instructions.")
    edit_strategy: Optional[NonEmptyStr] = Field(default=None, description="Editing to apply to OTHER vehicles only (add/modify/delete a vehicle, change its trajectory). Set to null if no other-vehicle editing is needed. Do NOT edit ego's trajectory here.")


class Response(ToolModel):
    scene_requirement: NonEmptyStr = Field(description="Target scene = scene context + ego's WRONG behavior (the failure to reproduce). One short sentence.")
    target_effect: NonEmptyStr = Field(description="Ego's CORRECT behavior that would have avoided the failure. One short sentence.")
    candidates: list[Candidate] = Field(min_length=3, max_length=3, description="Three variants of the same failure mechanism in slightly different road contexts.")


class Output(Response):
    pass
