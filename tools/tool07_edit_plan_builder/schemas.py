"""Typed editing plan contracts and instance-ID grounding."""
from typing import Annotated, Any
from pydantic import Field, StrictInt, field_validator
from tools.llm_runtime import NonEmptyStr, ToolModel, unwrap_result

InstanceId = StrictInt | NonEmptyStr
EditRequest = NonEmptyStr | Annotated[list[dict[str, Any] | NonEmptyStr], Field(min_length=1)]


class Input(ToolModel):
    edit_request: EditRequest = Field(description="Required natural-language edit request or non-empty list of editing ideas, supplied under edit_request.")
    scene_info: dict[str, Any] | str | None = Field(default=None, description="Scene context, including frame count, coordinates and constraints.")
    instances: list[dict[str, Any] | InstanceId] | dict[str, Any] = Field(default_factory=list, description="Actual discovered instances or the successful instance_discovery result/envelope. IDs must come from this context.")

    @field_validator("instances", mode="before")
    @classmethod
    def normalize_instances(cls,value):
        value=unwrap_result(value)
        if isinstance(value, dict) and "instances" in value:
            return value["instances"]
        return value


class Operation(ToolModel):
    operation: NonEmptyStr = Field(description="Named proposed edit operation.")
    target: InstanceId | None = Field(default=None, description="A supplied instance ID, or null when not yet known. Role descriptions belong in parameters.target_role.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Proposed edit parameters; distinguish design targets from measurements.")


class Response(ToolModel):
    edit_plan: NonEmptyStr = Field(description="Precise natural-language plan for SmartEdit; not its executable JSON config.")
    target_instances: list[InstanceId] = Field(default_factory=list, description="Only instance IDs actually present in the input context.")
    operations: Annotated[list[Operation], Field(min_length=1)] = Field(description="Concrete proposed operations.")
    constraints: list[NonEmptyStr] = Field(default_factory=list, description="Geometry, motion and evidence constraints.")


Output = Response


def known_instance_ids(data: Input) -> dict[str, str | int]:
    known={}
    def visit(value):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (str,int)) and not isinstance(item,bool):
                    known[str(item)]=item
                elif isinstance(item,dict):
                    visit(item)
        elif isinstance(value,dict):
            # Inspect object IDs, never IDs buried in per-frame measurements.
            for key in ("instance_id","object_id","id"):
                item=value.get(key)
                if isinstance(item,(str,int)) and not isinstance(item,bool):
                    known[str(item)]=item
            for key in ("instances","objects","result"):
                if isinstance(value.get(key),(list,dict)):
                    visit(value[key])
    visit(data.instances)
    if isinstance(data.scene_info,dict):
        visit(data.scene_info.get("instances",[]))
    return known


def validate_grounding(data: Input, output: Output) -> None:
    known=known_instance_ids(data)
    targets=list(output.target_instances)
    targets.extend(op.target for op in output.operations if op.target is not None)
    if any(str(target) not in known for target in targets):
        raise ValueError("Edit plan references an instance ID absent from the supplied instances; use null/empty targets when IDs are unknown")
