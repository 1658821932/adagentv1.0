"""Tool03 retrieval -> coarse plan -> validated tool05 arguments."""
from typing import Annotated, Any, Literal
from pydantic import Field, PrivateAttr, RootModel, model_validator
from common.scene_refs import SceneReference
from common.models import OutputModel
from common.artifacts import load_tool_result
from mcp_servers.scene_editing.adapter.schemas import AttackInput, VehicleAsset, AttackPlanOption, EgoHistoryRoute
from tools.llm_runtime import ToolModel, NonEmptyStr

class RetrievalContext(OutputModel):
    scene_ref: SceneReference
    video_path: NonEmptyStr | None = None
    source_file: NonEmptyStr | None = None
    source_segment: dict[str, float] | None = None
    mode: str | None = None


def normalize_retrieval(value):
    from pathlib import Path
    trusted=False
    if isinstance(value,str):
        value=load_tool_result(value,'video_retriever');trusted=True
    if isinstance(value,dict):
        if 'tool' in value or ('status' in value and 'result' in value):
            if value.get('tool')!='video_retriever' or value.get('status')!='success':
                raise ValueError('Expected a successful video_retriever envelope')
            value=value.get('result');trusted=True
        if not trusted:
            scene=value.get('scene_ref')
            if isinstance(scene,dict):
                missing=[k for k in ('frames_path','cameras_path','scenario_pkl')
                         if isinstance(scene.get(k),str) and not Path(scene[k]).exists()]
                if missing:
                    raise ValueError(
                        'retrieval_result scene_ref paths do not exist on disk: '
                        +', '.join(f'{k}={scene[k]}' for k in missing)
                        +'. Do not retype the retrieval JSON by hand: pass the video_retriever result_ref string from history verbatim as retrieval_result.')
    return value

class Input(ToolModel):
    retrieval_result: RetrievalContext = Field(
        description="Tool03 successful envelope, result object or absolute result_ref. Standalone scene_ref inputs are not accepted.")
    scene_requirement: NonEmptyStr = Field(description="Target scene from tool02.")
    target_effect: NonEmptyStr = Field(description="Unchanged global correct ego behavior from tool02.")
    edit_strategy: NonEmptyStr | None = Field(default=None, description="Selected tool02 candidate editing idea; null means no other-vehicle editing needed.")
    ego_input_edit_strategy: NonEmptyStr | None = Field(default=None, description='Optional intended change to ego history through 1.5s. Used only to plan stage 05A.')
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

class EgoHistoryPlan(ToolModel):
    mode: Literal['keep', 'speed', 'route']
    target_speed_mps: Annotated[float, Field(ge=0)] | None = None
    route: EgoHistoryRoute | None = None

    @model_validator(mode='after')
    def for_mode(self):
        if (self.target_speed_mps is not None) != (self.mode == 'speed') or (self.route is not None) != (self.mode == 'route'):
            raise ValueError('keep takes no parameters, speed takes target_speed_mps, and route takes route')
        return self

    def attack_arguments(self):
        if self.mode == 'keep':
            return {'prepare_ego_history': False}
        if self.mode == 'speed':
            return {'prepare_ego_history': True, 'ego_history_target_speed_mps': self.target_speed_mps}
        return {'prepare_ego_history': True, 'ego_history_route': self.route.model_dump(mode='json', exclude_none=True)}


class Response(ToolModel):
    suitability: Literal['ready','unsuitable','needs_information']
    reasons: list[NonEmptyStr] = Field(min_length=1)
    edit_plan: EditPlan | None = None
    edit_options: list[AttackPlanOption] = Field(default_factory=list,max_length=3,description="1..3 unranked coarse alternatives when editing is needed. Consider existing and added vehicles together; do not force a count. Legacy edit_plan is accepted only when edit_options is empty.")
    ego_history_plan: EgoHistoryPlan | None = Field(default=None, description='One shared 05A decision before all other-vehicle options. Choose keep, an ego speed target, or a reachable ego route endpoint at 1.5s.')
    retrieval_suggestion: NonEmptyStr | None = None

    @model_validator(mode='after')
    def decision(self):
        if self.edit_plan is not None and self.edit_options:
            raise ValueError('Use edit_options or legacy edit_plan, not both')
        ids=[option.option_id for option in self.edit_options]
        if len(ids)!=len(set(ids)):raise ValueError('edit_options option_id must be unique')
        # ready with edit_plan=null is allowed when no attack-vehicle editing is needed
        if self.suitability=='unsuitable' and (self.edit_plan is not None or self.edit_options or not self.retrieval_suggestion):
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
        if self.ego_history_plan is not None and self.attack_input is not None:
            for key, value in self.ego_history_plan.attack_arguments().items():
                if self.attack_input.get(key) != value:
                    raise ValueError('Stage 05A arguments differ from the reviewed ego_history_plan')
        if self.suitability=='ready':
            if self.edit_options:
                if self.attack_input is None:raise ValueError('Ready options require attack_input')
                attack=AttackInput.model_validate(self.attack_input)
                if (attack.scene_ref.scene_id,attack.scene_ref.scene_token)!=(self.scene_id,self.scene_token):
                    raise ValueError('Attack arguments belong to a different scene')
                if attack.operation!='auto' or attack.plan_options!=self.edit_options:
                    raise ValueError('Attack auto options differ from the reviewed alternatives')
            elif self.edit_plan is None:
                # No attack-vehicle editing needed; scene is ready for ego editing directly
                if self.attack_input is not None:
                    raise ValueError('attack_input must be null when edit_plan is null')
            else:
                if self.attack_input is None:raise ValueError('Ready with edit_plan requires validated attack_input')
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

class TrajectoryRequest(ToolModel):
    action: Literal['request_trajectory']
    track_id: NonEmptyStr = Field(description="Exact vehicle track ID from the catalog, or ego for the recorded ego reference.")
    max_samples: int = Field(default=24,ge=2,le=24)
    purpose: NonEmptyStr

class Decision(RootModel[Response | ViewRequest | TrajectoryRequest]):
    """Internal visual request or final review; public Output stays unchanged."""
