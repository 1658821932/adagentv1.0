"""Structured generation cannot change policy/history or replenish shared budgets."""
import importlib
import json
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest

from mcp_servers.scene_editing.core.adversarial.defense_generation_contract import validate_request, select_foci_parent

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


def request(**changes):
    result=dict(parent_trajectory_id='recorded_baseline',action='reduce_speed',
        start_time_range_s=[4.,5.],target_speed_mps=2.,deceleration_range_mps2=[1.,3.],
        end_behavior='cruise',max_candidates=2)
    result.update(changes)
    return result


@pytest.mark.parametrize('changes', [dict(start_time_range_s=[2.9,5.]),dict(start_time_range_s=[4.,12.]),
    dict(target_speed_mps=float('nan')),dict(max_candidates=True),dict(max_candidates=3),
    dict(target_speed_mps=0.,end_behavior='cruise'),dict(deceleration_range_mps2=[0.,2.]),
    dict(acceptance_policy={'max_acceleration_mps2':999})])
def test_reject_history_budget_nonfinite_and_policy_override(changes):
    with pytest.raises(ValueError):
        validate_request('construct_defense_candidates',request(**changes),boundary=3.,end=11.5,remaining=2)


def test_foci_parent_is_not_selected_by_the_largest_gap():
    rows=[dict(candidate_id='failed',reference_screen_passed=False,shared_checks={'acceleration':False},
               max_acceleration_mps2=3.,generation_curve={'max_acceleration_mps2':10.},min_target_clearance_m=30.),
          dict(candidate_id='strong',reference_screen_passed=True,shared_checks={'acceleration':True},
               max_acceleration_mps2=4.5,min_target_clearance_m=30.),
          dict(candidate_id='gentle',reference_screen_passed=True,shared_checks={'acceleration':True},
               max_acceleration_mps2=2.,min_target_clearance_m=7.)]
    assert select_foci_parent(rows)=='gentle'
    assert select_foci_parent([]) is None


def generator(monkeypatch,tmp_path,invalid_first=False,**options):
    monkeypatch.syspath_prepend(str(CORE))
    runtime=types.ModuleType('experiment_runtime')
    runtime.atomic_json=lambda path,value:Path(path).write_text(json.dumps(value))
    monkeypatch.setitem(__import__('sys').modules,'experiment_runtime',runtime)
    spec=importlib.util.spec_from_file_location('generation_budget_test',CORE/'defense_generation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cls=module.DefenseGeneration
    xy=np.array([[0.,0.],[1.,0.],[2.,0.]])
    editor=types.SimpleNamespace(run_dir=tmp_path,input_dir=tmp_path,target_track='target',
        baseline_xy=xy,baseline_yaw=np.zeros(3),times=np.array([0.,3.,11.5]),edit_count=0,
        effective_policy=dict(observation_end_s=3.,max_speed_mps=15.,max_acceleration_mps2=6.),
        nodes={'baseline':dict(xy=xy,yaw=np.zeros(3))})
    calls=[]
    def edit(args):
        calls.append(args)
        if invalid_first and len(calls)==1:
            return dict(status='invalid',error='Requested transition exceeds scene duration')
        editor.edit_count+=1;cid=f'edit_{editor.edit_count:03d}'
        editor.nodes[cid]=dict(xy=xy,yaw=np.zeros(3))
        return dict(status='reference_screen_failed',candidate_id=cid)
    editor.edit=edit
    editor._public=lambda cid:dict(candidate_id=cid,status='reference_screen_failed',failed_checks=['target_clearance'])
    return cls(editor,2,**options),calls


def test_failure_candidates_and_rejected_attempts_are_retained_and_charged(monkeypatch,tmp_path):
    gen,calls=generator(monkeypatch,tmp_path,invalid_first=True)
    result=gen.execute('construct_defense_candidates',request())
    assert result['status']=='candidates_generated' and len(result['candidates'])==1
    assert result['candidates'][0]['failed_checks']==['target_clearance']
    assert len(result['construction_failures'])==1 and len(calls)==2
    assert gen.remaining()==0 and gen.tools()==[]
    assert all(c['generation_source']=='constructor' for c in calls)
    saved=json.loads(gen.path.read_text())
    assert saved['rejected_attempts']==1 and saved['calls'][0]['status']=='completed'
    restored=type(gen)(gen.editor,2)
    assert restored.context()==gen.context()
    with pytest.raises(ValueError):restored.execute('construct_defense_candidates',request(max_candidates=1))


def test_regular_edits_consume_the_same_candidate_budget(monkeypatch,tmp_path):
    gen,_=generator(monkeypatch,tmp_path)
    gen.editor.edit_count=2
    assert gen.remaining()==0 and gen.tools()==[]


def test_disabled_foci_is_not_offered_or_executed(monkeypatch,tmp_path):
    gen,_=generator(monkeypatch,tmp_path,foci_enabled=False)
    assert [t['function']['name'] for t in gen.tools()]==['construct_defense_candidates']
    with pytest.raises(ValueError,match='disabled'):
        gen.execute('optimize_defense_candidate',dict(parent_candidate_id='baseline',edit_window_s=[3.,10.],max_candidates=1))
    assert gen.state['calls']==[]


def test_foci_timeout_is_execution_error_and_preserves_consumed_call(monkeypatch,tmp_path):
    gen,_=generator(monkeypatch,tmp_path)
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired('foci',300)
    monkeypatch.setattr(subprocess,'run',timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        gen.execute('optimize_defense_candidate',dict(parent_candidate_id='baseline',edit_window_s=[4.,10.],max_candidates=1))
    state=json.loads(gen.path.read_text())
    assert state['foci_calls']==1 and state['calls'][0]['status']=='execution_failed'
    assert gen.editor.edit_count==0


def test_hard_policy_cannot_be_raised_by_search_ranges(monkeypatch,tmp_path):
    gen,calls=generator(monkeypatch,tmp_path)
    with pytest.raises(ValueError,match='unchanged research policy'):
        gen.execute('construct_defense_candidates',request(deceleration_range_mps2=[2.,7.]))
    assert not calls and gen.state['calls']==[]


@pytest.mark.parametrize('has_candidates',[True,False])
def test_default_foci_stage_reports_normal_empty_pool_and_correct_window(monkeypatch,tmp_path,has_candidates):
    gen,_=generator(monkeypatch,tmp_path)
    # Use the module of the loaded production class, which is deliberately isolated.
    globals_=gen.__class__.__init__.__globals__
    function=globals_['prepare_initial_foci']
    constructor=tmp_path/'constructor';(constructor/'reference_screen').mkdir(parents=True)
    rows=[dict(candidate_id='failed',reference_screen_passed=False,shared_checks={'acceleration':False},
               max_acceleration_mps2=10.,min_target_clearance_m=30.),
          dict(candidate_id='gentle',reference_screen_passed=True,shared_checks={'acceleration':True},
               max_acceleration_mps2=2.,min_target_clearance_m=7.)]
    (constructor/'reference_screen/summary.json').write_text(json.dumps(dict(candidates=rows,target_track='target',acceptance_policy=dict(observation_end_s=3.))))
    commands=[]
    def execute(command,**kwargs):
        commands.append(command);out=Path(command[command.index('--run')+1]);out.mkdir()
        records=[dict(candidate_id='foci_001')] if has_candidates else []
        (out/'summary.json').write_text(json.dumps(dict(candidates=records,gaussian_optimizer='executed' if has_candidates else 'skipped_initial_overlap_not_finite_positive')))
    monkeypatch.setattr(subprocess,'run',execute)
    out=tmp_path/'default_foci'
    result=function(tmp_path,constructor,'target',out,3.,11.5)
    assert result==(out if has_candidates else None)
    command=commands[0]
    assert command[command.index('--candidate')+1]=='gentle'
    assert command[command.index('--edit-window')+1:command.index('--edit-window')+3]==['3.0','11.5']
    stage=json.loads((tmp_path/'foci-stage.json').read_text())
    assert stage['fallback_to_constructor'] is not has_candidates
    assert stage['status']==('completed' if has_candidates else 'no_candidates')


def test_default_foci_execution_error_does_not_become_empty_pool(monkeypatch,tmp_path):
    gen,_=generator(monkeypatch,tmp_path)
    function=gen.__class__.__init__.__globals__['prepare_initial_foci']
    constructor=tmp_path/'constructor';(constructor/'reference_screen').mkdir(parents=True)
    (constructor/'reference_screen/summary.json').write_text(json.dumps(dict(candidates=[dict(candidate_id='a',reference_screen_passed=True,max_acceleration_mps2=2.)],target_track='target',acceptance_policy=dict(observation_end_s=3.))))
    def broken(*args,**kwargs):raise subprocess.CalledProcessError(1,'foci')
    monkeypatch.setattr(subprocess,'run',broken)
    with pytest.raises(subprocess.CalledProcessError):function(tmp_path,constructor,'target',tmp_path/'foci',3.,11.5)
    assert json.loads((tmp_path/'foci-stage.json').read_text())['status']=='execution_failed'


def test_rounded_endpoint_is_normalized_without_touching_history_or_input():
    from mcp_servers.scene_editing.core.adversarial.defense_generation_contract import normalize_request
    original=dict(parent_candidate_id='a',edit_window_s=[3.,11.5],max_candidates=1)
    normalized=normalize_request('optimize_defense_candidate',original,11.499597)
    assert normalized['edit_window_s']==[3.,11.499597]
    assert original['edit_window_s']==[3.,11.5]
    validate_request('optimize_defense_candidate',normalized,boundary=3.,end=11.499597,remaining=1)
    bad=normalize_request('optimize_defense_candidate',dict(original,edit_window_s=[2.9999,11.5]),11.499597)
    with pytest.raises(ValueError):validate_request('optimize_defense_candidate',bad,boundary=3.,end=11.499597,remaining=1)


def test_foci_explicit_goals_are_optional_and_do_not_relax_history_or_policy():
    args=dict(parent_candidate_id='p',edit_window_s=[3.,11.5],max_candidates=1,
              preserve_route=True,terminal_behavior='stop',improvement_goal='smoother_motion')
    assert validate_request('optimize_defense_candidate',args,boundary=3.,end=11.5,remaining=1)==1
    for changes in (dict(preserve_route='yes'),dict(terminal_behavior='cruise'),
                    dict(edit_window_s=[3.,10.]),dict(acceptance_policy={})):
        with pytest.raises(ValueError):
            validate_request('optimize_defense_candidate',dict(args,**changes),boundary=3.,end=11.5,remaining=1)
