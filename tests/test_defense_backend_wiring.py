"""The registered ego backend forwards source images, goals and the FOCI stage."""
import ast
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


@pytest.mark.parametrize('enabled,has_pool',[(True,True),(True,False),(False,False)])
def test_backend_images_goal_and_foci_fallback_reach_editor(monkeypatch,tmp_path,enabled,has_pool):
    scene=tmp_path/'frozen';(scene/'metadata').mkdir(parents=True)
    (scene/'metadata/frames.json').write_text('[]')
    attack=tmp_path/'attack';attack.mkdir()
    (attack/'input.json').write_text(json.dumps(dict(constraints=dict(acceptance_policy=dict(observation_end_s=3.)))))
    output=tmp_path/'output';output.mkdir();observed={}
    class ReachedEditor(Exception):pass
    def atomic(path,value):
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value))
    def understand(*args,**kwargs):
        observed['understanding']=kwargs;atomic(args[4]/'summary.json',dict(status='completed'))
    def foci(*args,**kwargs):
        observed['foci']=(args,kwargs)
        return args[3] if has_pool else None
    def edit(*args,**kwargs):
        observed['editor']=(args,kwargs);raise ReachedEditor
    modules={
        'defense_probe_feedback':dict(reuse=lambda *args:False),
        'defense_generation':dict(prepare_initial_foci=foci),
        'defense_request_recovery':dict(run_with_one_retry=lambda fn,args,options,run_dir:fn(*args,**options))}
    for name,entries in modules.items():
        module=types.ModuleType(name)
        for key,value in entries.items():setattr(module,key,value)
        monkeypatch.setitem(sys.modules,name,module)
    defaults=dict(defense_constructor_candidates=24,defense_initial_foci_candidates=2,
        defense_max_turns=10,defense_max_edit_candidates=5,defense_max_inspections=3)
    namespace=dict(Path=Path,json=json,atomic_json=atomic,policy=types.SimpleNamespace(resolve=lambda value:dict(value)),
        ego_path_from_frames=lambda frames:(np.array([0.,3.,11.5]),None,None),
        run_understanding=understand,run_constructor=lambda *args:None,run_defense_edit=edit,SETTINGS=dict(defaults=defaults))
    node=next(n for n in ast.parse((CORE/'editing_backend.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='ego_output')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(CORE/'editing_backend.py'),'exec'),namespace)
    args=dict(attack_result=dict(frozen_scene_path=str(scene),submission_path=str(attack/'trajectory_editing/output.json'),
        target_track='target',selected_candidate_id='attack_001',source_scene_ref=dict(sample_root='/original/scene')),
        allow_safe_stop=True,use_foci=enabled,target_effect='Use the requested maneuver and compare relevant alternatives.',timeout=3600)
    with pytest.raises(ReachedEditor):namespace['ego_output'](args,output)
    assert observed['understanding']['image_scene']=='/original/scene'
    assert observed['understanding']['target_effect']==args['target_effect']
    positional,options=observed['editor']
    assert positional[2]==(output/'defense_foci' if enabled and has_pool else None)
    assert options['controlled_generation'] and options['foci_enabled'] is enabled
    assert options['target_effect']==args['target_effect']
    assert ('foci' in observed) is enabled
