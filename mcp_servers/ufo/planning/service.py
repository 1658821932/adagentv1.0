"""UFO planning using the main branch typed LLM contract and bounded retries."""
from pathlib import Path
from typing import Annotated, Literal
from pydantic import Field, model_validator
from tools.llm_runtime import LLMTool, NonEmptyStr, ToolModel
from tools.tool07_edit_plan_builder.schemas import Input as BaseInput, Response as BaseResponse, Operation
from tools.tool05_instance_discovery.selection import load_bound_instances
from mcp_servers.ufo.metric_config import validate

class Input(BaseInput):
    backend: Literal['ufo'] = 'ufo'
    operation: Literal['lane_change', 'metric_edit'] = 'lane_change'
    calibration_path: NonEmptyStr | None = None
    asset_manifest_path: NonEmptyStr | None = None
    target_selection_path: NonEmptyStr | None = None

    @model_validator(mode='after')
    def bind_target(self):
        if self.target_selection_path:
            _, selected = load_bound_instances(self.target_selection_path)
            self.instances = [selected]
        elif self.operation == 'lane_change':
            raise ValueError('UFO lane planning requires target_selection_path')
        if self.operation == 'metric_edit' and not self.calibration_path:
            raise ValueError('Metric planning requires calibration_path')
        return self

class Insert(ToolModel):
    operation: Literal['insert']
    object_id: NonEmptyStr
    asset_manifest_path: NonEmptyStr
    position_m: Annotated[list[float], Field(min_length=3,max_length=3)]
    yaw_degrees: float = 0
    motion_reference: Literal['world_fixed','ego_projected','ego_translation'] = 'world_fixed'

class Translate(ToolModel):
    operation: Literal['translate']
    target_id: NonEmptyStr
    offset_m: Annotated[list[float], Field(min_length=3,max_length=3)]
    start_seconds: float
    end_seconds: float
    trajectory_mode: Literal['offset_from_original'] = 'offset_from_original'

class Response(BaseResponse):
    operations: Annotated[list[Insert | Translate | Operation], Field(min_length=1)]

class Output(Response):
    target_selection_path: str | None = None
    selected_instance_info_path: str | None = None

def enrich(data, result):
    if data.target_selection_path:
        binding, _ = load_bound_instances(data.target_selection_path)
        result.update(target_selection_path=data.target_selection_path,
                      selected_instance_info_path=binding['selected_instance_info_path'])
    return result

def check(data, output):
    if data.target_selection_path:
        binding, _ = load_bound_instances(data.target_selection_path)
        if output.target_instances != [binding['selected_instance_id']]:
            raise ValueError('Edit plan changed the selected target')
    elif output.target_instances:
        raise ValueError('Insert-only plan cannot invent detected target IDs')
    if data.operation == 'metric_edit':
        config = validate(dict(backend='ufo',operation='metric_edit',units='meters',
            calibration_path=data.calibration_path,target_selection_path=data.target_selection_path,
            operations=[op.model_dump(mode='json') for op in output.operations]),agent=True)
        for op in config['operations']:
            if op['operation'] == 'insert' and op['asset_manifest_path'] != data.asset_manifest_path:
                raise ValueError('Plan changed the supplied asset manifest')
    else:
        ident = output.target_instances[0]
        if any(not isinstance(op, Operation) or type(op.target) is not int or op.target != ident for op in output.operations):
            raise ValueError('Visual plan changed the selected target')

TOOL = LLMTool(name='edit_plan_builder',directory=Path(__file__).parent,
    input_model=Input,response_model=Response,output_model=Output,
    enrich_result=enrich,validate_result=check)

def run(input_data):
    return TOOL.execute(input_data)
