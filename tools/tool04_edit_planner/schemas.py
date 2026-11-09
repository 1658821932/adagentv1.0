"""Tool03 retrieval -> coarse plan -> validated tool05 arguments."""
from typing import Any, Literal
from pydantic import Field, PrivateAttr, RootModel, model_validator
from common.scene_refs import SceneReference
from common.models import OutputModel
from common.artifacts import load_tool_result
from mcp_servers.scene_editing.adapter.schemas import AttackInput, VehicleAsset
from tools.llm_runtime import ToolModel, NonEmptyStr

class RetrievalContext(OutputModel):
    scene_ref: SceneReference
    video_path: NonEmptyStr | None = None
    source_file: NonEmptyStr | None = None
    source_segment: dict[str, float] | None = None
    mode: str | None = None


def normalize_retrieval(value):
    if isinstance(value,str):value=load_tool_result(value,'video_retriever')
    if isinstance(value,dict):
        if 'tool' in value or ('status' in value and 'result' in value):
            if value.get('tool')!='video_retriever' or value.get('status')!='success':
                raise ValueError('Expected a successful video_retriever envelope')
            value=value.get('result')
    return value

class Input(ToolModel):
    retrieval_result: RetrievalContext = Field(
        description="Tool03 successful envelope, result object or absolute result_ref. Standalone scene_ref inputs are not accepted.")
    scene_requirement: NonEmptyStr = Field(description="Target scene from tool02.")
    target_effect: NonEmptyStr = Field(description="Unchanged global correct ego behavior from tool02.")
    edit_strategy: NonEmptyStr = Field(description="Selected tool02 candidate editing idea.")
    vehicle: VehicleAsset | None = Field(default=None,description="Optional real added-vehicle PLY and length/width/height. Required for executable add_vehicle arguments; do not infer from images.")

    _retrieval_result_ref: str | None = PrivateAttr(default=None)

    @model_validator(mode='wrap')
    @classmethod
    def preserve_result_ref(cls,value,handler):
        original=value.get('retrieval_result') if isinstance(value,dict) else None
        result=handler(value)
        if isinstance(original,str):result._retrieval_result_ref=original
        return result

    @property
    def retrieval_result_ref(self):return self._retrieval_result_ref

    @model_validator(mode='before')
    @classmethod
    def resolve_reference(cls,value):
        if isinstance(value,dict):
            value=dict(value)
            if 'retrieval_result' in value:
                value['retrieval_result']=normalize_retrieval(value['retrieval_result'])
        return value

    @property
    def scene_ref(self):return self.retrieval_result.scene_ref

class EditPlan(ToolModel):
    operation: Literal['edit_existing','add_vehicle']
    target_track: NonEmptyStr | None = None
    maneuver: Literal['stationary','left_cut_in'] | None = None
    instruction: NonEmptyStr = Field(description="Scene-specific coarse vehicle behavior and event order; detailed trajectories are searched by tool05.")

    @model_validator(mode='after')
    def target(self):
        if self.operation=='edit_existing' and (self.target_track is None or self.maneuver is not None):
            raise ValueError('Existing-vehicle plan requires annotated target_track and no add-vehicle maneuver')
        if self.operation=='add_vehicle' and (self.target_track is not None or self.maneuver is None):
            raise ValueError('New-vehicle plan requires a supported maneuver and no invented track ID')
        return self

class Response(ToolModel):
    suitability: Literal['ready','unsuitable','needs_information']
    reasons: list[NonEmptyStr] = Field(min_length=1)
    edit_plan: EditPlan | None = None
    retrieval_suggestion: NonEmptyStr | None = None

    @model_validator(mode='after')
    def decision(self):
        if self.suitability=='ready' and self.edit_plan is None:
            raise ValueError('Ready decisions require a coarse edit plan')
        if self.suitability=='unsuitable' and (self.edit_plan is not None or not self.retrieval_suggestion):
            raise ValueError('Unsuitable scenes require a retrieval suggestion and no edit plan')
        return self

class Output(Response):
    scene_id: NonEmptyStr
    scene_token: NonEmptyStr
    target_effect: NonEmptyStr
    evidence: dict
    attack_input: dict[str,Any] | None = Field(default=None,description="Program-built tool05 arguments validated against AttackInput. Forward this object unchanged when ready; null otherwise.")

    @model_validator(mode='after')
    def handoff(self):
        if self.suitability=='ready':
            if self.attack_input is None:raise ValueError('Ready output requires validated attack_input')
            attack=AttackInput.model_validate(self.attack_input)
            if (attack.scene_ref.scene_id,attack.scene_ref.scene_token)!=(self.scene_id,self.scene_token):
                raise ValueError('Attack arguments belong to a different scene')
            plan=self.edit_plan
            if (attack.operation,attack.target_track,attack.maneuver,attack.instruction)!=(plan.operation,plan.target_track,plan.maneuver,plan.instruction):
                raise ValueError('Attack arguments differ from the reviewed plan')
        elif self.attack_input is not None:
            raise ValueError('Non-ready results cannot carry executable attack arguments')
        return self


class ViewRequest(ToolModel):
    action: Literal['request_views']
    cameras: list[NonEmptyStr] = Field(min_length=1,max_length=2,description="Actual camera names from the source scene, e.g. CAM_L0 or CAM_R0.")
    purpose: NonEmptyStr = Field(description="What scene requirement or uncertainty needs these additional views.")

class Decision(RootModel[Response | ViewRequest]):
    """Internal visual request or final review; public Output stays unchanged."""
