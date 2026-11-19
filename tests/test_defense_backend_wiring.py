"""The registered ego backend forwards source images, goals and the FOCI stage."""
import ast
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
if str(CORE) not in sys.path:
    sys.path.insert(0,str(CORE))


@pytest.mark.parametrize('enabled,has_pool,initial',[(True,True,False),(True,False,False),(False,False,False),(True,True,True)])
def test_backend_images_goal_and_on_demand_foci_reach_editor(monkeypatch,tmp_path,enabled,has_pool,initial):
    scene=tmp_path/'frozen';(scene/'metadata').mkdir(parents=True)
    (scene/'metadata/frames.json').write_text('[]')
    attack=tmp_path/'attack';attack.mkdir()
    (attack/'input.json').write_text(json.dumps(dict(constraints=dict(acceptance_policy=dict(observation_end_s=3.)))))
    output=tmp_path/'output';output.mkdir();observed={};calls={'understanding':0,'constructor':0,'foci':0}
    from mcp_servers.scene_editing.core.adversarial import ego_stage_reuse
    monkeypatch.setitem(sys.modules,'ego_stage_reuse',ego_stage_reuse)
    class ReachedEditor(Exception):pass
    def atomic(path,value):
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value))
    def understand(*args,**kwargs):
        calls['understanding']+=1
        observed['understanding']=kwargs;atomic(args[4]/'summary.json',dict(status='completed'))
        atomic(args[4]/'config.json',dict(scene_dir=str(scene),target_track='target'))
        atomic(args[4]/'conflict-brief.json',dict(directions=[]))
    def foci(*args,**kwargs):
        calls['foci']+=1
        observed['foci']=(args,kwargs)
        atomic(args[3].parent/'foci-stage.json',dict(status='completed' if has_pool else 'no_candidates'))
        atomic(args[3]/'summary.json',dict(candidates=[]))
        return args[3] if has_pool else None
    def constructor(*args,**kwargs):
        calls['constructor']+=1
        atomic(args[2]/'reference_screen/summary.json',dict(status='reference_screen_complete',candidates=[]))
    def edit(*args,**kwargs):
        observed['editor']=(args,kwargs);raise ReachedEditor
    modules={
        'defense_probe_feedback':dict(reuse=lambda *args:False),
        'defense_generation':dict(prepare_initial_foci=foci,
            should_prepare_initial_foci=lambda config,default=False:bool(config.get('use_foci',True) and config.get('initial_foci',default))),
        'defense_request_recovery':dict(run_with_one_retry=lambda fn,args,options,run_dir:fn(*args,**options))}
    for name,entries in modules.items():
        module=types.ModuleType(name)
        for key,value in entries.items():setattr(module,key,value)
        monkeypatch.setitem(sys.modules,name,module)
    defaults=dict(defense_constructor_candidates=24,defense_initial_foci_candidates=2,
        defense_initial_foci_enabled=initial,defense_max_turns=10,defense_max_edit_candidates=5,defense_max_inspections=3)
    namespace=dict(Path=Path,json=json,atomic_json=atomic,policy=types.SimpleNamespace(resolve=lambda value:dict(value),for_new_run=lambda value:dict(value),edit_start=lambda value:value.get('edit_start_s',value['observation_end_s']),resolve_ego_progress_policy=__import__('acceptance_policy').resolve_ego_progress_policy),
        ego_path_from_frames=lambda frames:(np.r_[0., .5003, 1.0006, 1.500528, np.arange(2., 12., .5)],None,None),
        run_understanding=understand,run_constructor=constructor,run_defense_edit=edit,SETTINGS=dict(defaults=defaults))
    node=next(n for n in ast.parse((CORE/'editing_backend.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='ego_output')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(CORE/'editing_backend.py'),'exec'),namespace)
    args=dict(attack_result=dict(frozen_scene_path=str(scene),submission_path=str(attack/'trajectory_editing/output.json'),
        target_track='target',selected_candidate_id='attack_001',source_scene_ref=dict(sample_root='/original/scene')),
        allow_safe_stop=True,use_foci=enabled,target_effect='Use the requested maneuver and compare relevant alternatives.',timeout=3600)
    with pytest.raises(ReachedEditor):namespace['ego_output'](args,output)
    assert observed['understanding']['image_scene']=='/original/scene'
    assert observed['understanding']['target_effect']==args['target_effect']
    assert observed['understanding']['visual_frame'] == 3
    assert observed['understanding']['acceptance']['edit_start_s'] == 1.500528
    assert observed['understanding']['acceptance']['observation_end_s'] == 1.500528
    positional,options=observed['editor']
    prepare=enabled and initial
    assert positional[2]==(output/'defense_foci' if prepare and has_pool else None)
    assert options['controlled_generation'] and options['foci_enabled'] is enabled
    assert options['target_effect']==args['target_effect']
    assert ('foci' in observed) is prepare
    if enabled and not initial:
        assert json.loads((output/'foci-stage.json').read_text())['status']=='on_demand'

    assert calls==dict(understanding=1,constructor=1,foci=1 if prepare else 0)
    atomic(tmp_path/'input/stage_request.json',dict(stage='ego_editing',arguments=args))
    next_output=tmp_path/'retry/output';next_output.mkdir(parents=True)
    retry_args=dict(args,reuse_completed_from=str(tmp_path))
    with pytest.raises(ReachedEditor):namespace['ego_output'](retry_args,next_output)
    assert calls==dict(understanding=1,constructor=1,foci=1 if prepare else 0)
    positional,options=observed['editor']
    assert positional[1]==next_output/'defense_constructor'
    assert positional[2]==(next_output/'defense_foci' if prepare and has_pool else None)
    assert positional[3]==next_output/'defense_agent'
    for stage_name in ['defense_understanding','defense_constructor']+(['defense_foci'] if prepare else []):
        provenance=json.loads((next_output/stage_name/'reuse-provenance.json').read_text())
        assert provenance['status']=='reused'



def test_constructor_timeout_returns_recoverable_result_and_reuses_understanding(monkeypatch,tmp_path):
    import time
    scene=tmp_path/'frozen';(scene/'metadata').mkdir(parents=True)
    (scene/'metadata/frames.json').write_text('[]')
    attack=tmp_path/'attack';attack.mkdir()
    (attack/'input.json').write_text(json.dumps(dict(constraints=dict(acceptance_policy=dict(observation_end_s=3.)))))
    output=tmp_path/'output';output.mkdir();calls=dict(understanding=0,constructor=0)
    from mcp_servers.scene_editing.core.adversarial import ego_stage_reuse
    monkeypatch.setitem(sys.modules,'ego_stage_reuse',ego_stage_reuse)
    class ReachedEditor(Exception):pass
    def atomic(path,value):
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value))
    def understand(*args,**kwargs):
        calls['understanding']+=1
        atomic(args[4]/'summary.json',dict(status='completed'))
        atomic(args[4]/'config.json',dict(scene_dir=str(scene),target_track='target'))
        atomic(args[4]/'conflict-brief.json',dict(directions=[]))
    state=dict(timeout=True)
    def constructor(*args,**kwargs):
        calls['constructor']+=1
        if state['timeout']:
            time.sleep(.2)
            return
        atomic(args[2]/'reference_screen/summary.json',dict(status='reference_screen_complete',candidates=[]))
    def edit(*args,**kwargs):raise ReachedEditor
    modules={
        'defense_probe_feedback':dict(reuse=lambda *args:False),
        'defense_generation':dict(prepare_initial_foci=lambda *args:None,
            should_prepare_initial_foci=lambda config,default=False:False),
        'defense_request_recovery':dict(run_with_one_retry=lambda fn,args,options,run_dir:fn(*args,**options))}
    for name,entries in modules.items():
        module=types.ModuleType(name)
        for key,value in entries.items():setattr(module,key,value)
        monkeypatch.setitem(sys.modules,name,module)
    defaults=dict(defense_constructor_candidates=24,defense_initial_foci_candidates=2,
        defense_max_turns=10,defense_max_edit_candidates=5,defense_max_inspections=3,
        defense_preparation_timeout_seconds=.1)
    namespace=dict(Path=Path,json=json,atomic_json=atomic,policy=types.SimpleNamespace(resolve=lambda value:dict(value),for_new_run=lambda value:dict(value),edit_start=lambda value:value.get('edit_start_s',value['observation_end_s']),resolve_ego_progress_policy=__import__('acceptance_policy').resolve_ego_progress_policy),
        ego_path_from_frames=lambda frames:(np.arange(0.,12.,.5),None,None),
        run_understanding=understand,run_constructor=constructor,run_defense_edit=edit,SETTINGS=dict(defaults=defaults))
    node=next(n for n in ast.parse((CORE/'editing_backend.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='ego_output')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(CORE/'editing_backend.py'),'exec'),namespace)
    args=dict(attack_result=dict(frozen_scene_path=str(scene),submission_path=str(attack/'trajectory_editing/output.json'),
        target_track='target',selected_candidate_id='attack_001',source_scene_ref=dict(sample_root='/original/scene')),
        allow_safe_stop=True,use_foci=False,target_effect='Stop safely',timeout=3600)
    atomic(tmp_path/'input/stage_request.json',dict(stage='ego_editing',arguments=args))
    result=namespace['ego_output'](args,output)
    assert result['status']=='defense_not_found' and result['stop_reason']=='preparation_timeout'
    assert result['detail']['preparation_stage']=='defense_constructor'
    assert result['run_dir']==str(output)
    assert (output/'defense_understanding/completed-stage.json').is_file()
    state['timeout']=False
    retry_output=tmp_path/'retry/output';retry_output.mkdir(parents=True)
    with pytest.raises(ReachedEditor):namespace['ego_output'](dict(args,reuse_completed_from=str(tmp_path)),retry_output)
    assert calls==dict(understanding=1,constructor=2)
    assert json.loads((retry_output/'defense_understanding/reuse-provenance.json').read_text())['status']=='reused'
