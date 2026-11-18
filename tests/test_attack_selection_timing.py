"""Regression: object selection's local `time` must not shadow its API timer."""
import ast
import json
from pathlib import Path
from time import perf_counter

import numpy as np


CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


def test_first_object_response_is_processed_before_prefix_loop_binds_time(tmp_path):
    scene=tmp_path/'scene';(scene/'metadata').mkdir(parents=True)
    (scene/'metadata/frames.json').write_text('[{}, {}, {}]')
    times=np.array([0.,.5,1.])
    ego=np.array([[0.,0.],[1.,0.],[2.,0.]])
    actors=[[dict(track='fixture-car',label='vehicle',x=10.,y=0.,yaw=0.)] for _ in times]
    call=dict(id='fixture',function=dict(name='finish_search',arguments='{"reason":"fixture"}'))
    exchanges=[]
    class Archive:
        def complete(self,*args,**kwargs):
            return '', 'fixture-attempt'
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    def exchange(*args,**kwargs):
        assert kwargs['started']<=perf_counter()
        exchanges.append((args,kwargs))
    namespace=dict(json=json,np=np,perf_counter=perf_counter,atomic_json=write,reasoning_effort=lambda:'low',
        ego_path_from_frames=lambda frames:(times,ego,np.zeros(3)),
        observed_objects_from_frames=lambda frames:actors,load_official_drivable=lambda path:None,
        _road_sections=lambda *args:[],EGO_FRONT=3.,EGO_REAR=1.,EGO_WIDTH=2.,
        OBJECT_TOOLS=[],placement_preview_tool=lambda:dict(type='function',function={'name':'preview_attack_options'}),
        OBJECT_PROMPT='fixture',compact_model_context=lambda value:value,image_part=lambda path:dict(path=str(path)),
        decode_call=lambda attempt:(call,'finish_search',dict(reason='fixture')),
        archive_exchange=exchange,revision_evidence=lambda constraints:{})
    node=next(n for n in ast.parse((CORE/'run_attack_workflow.py').read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='select_object')
    node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),'select_object','exec'),namespace)
    result=namespace['select_object'](scene,tmp_path/'selection','fixture',[],{},Archive(),10,128)
    assert result['stop_reason']=='finished_without_object_selection'
    assert len(exchanges)==1
