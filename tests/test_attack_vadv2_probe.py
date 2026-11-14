"""Bounded attack self-tests: no stale, untested or merely zero-score successes."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from subprocess import CompletedProcess
import sys
import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


def make_probe(tmp_path,monkeypatch,root=None):
    monkeypatch.syspath_prepend(str(CORE))
    from attack_policy_probe import VADProbe
    editor=SimpleNamespace(constraints=dict(vadv2_probe_root=str(root or tmp_path/'probes'),
        vadv2_probe_budget=3,policy_source_scene_ref={},vadv2_policy_config=dict(python=sys.executable,
            worldengine_root=str(tmp_path),config_path='/cfg',checkpoint_path='/weights')),
        scene=tmp_path/'scene',run_dir=tmp_path/'editor',times=np.arange(6)*.5,
        target=dict(track_id='target',operation='edit_existing'),nodes={})
    for key in ('a','b','c','d'):
        editor.nodes[key]=dict(xy=np.array([[0.,0.],[1.,0.]]),yaw=np.array([0.,0.]),
            result=dict(reference_screen_passed=True,construction_goal=dict(goal_met=True)))
    return VADProbe(editor)


def fake_native(probe,monkeypatch,collision=True,history=False):
    def materialize(candidate,folder):
        frozen=folder/'frozen';(frozen/'metadata').mkdir(parents=True)
        (frozen/'metadata/frames.json').write_text('[]')
        return frozen
    probe._materialize=materialize
    probe._plot=lambda collision,folder:folder/'overlay.png'
    monkeypatch.setitem(sys.modules,'native_scene_export',SimpleNamespace(export=Mock(return_value=dict(
        simulation_input=dict(scenario_data_folder_path='/package',asset_folder_path='/assets',scenario_pkl_file_name='all_scenarios.pkl')))))
    report=dict(status='completed',target_track='target',target_collision_detected=collision,
        first_target_collision_time_s=2 if collision else None,min_target_clearance_m=0 if collision else 3,
        closest_approach=dict(time_s=2,clearance_m=0 if collision else 3),
        initial_history_target_contact=history,possible_interpolated_target_contact=False,scope='actual test fixture')
    runner=Mock(return_value=CompletedProcess([],0,json.dumps(dict(status='success',result=dict(collision_report=report))),''))
    monkeypatch.setattr('mcp_servers.process.run_logged',runner)
    return runner


def test_untested_or_noncolliding_candidate_cannot_be_submitted(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='exact candidate'):probe.require_challenge('a')
    fake_native(probe,monkeypatch,collision=False)
    assert probe.probe('a')['policy_attack_success'] is False
    with pytest.raises(ValueError,match='not hit'):probe.require_challenge('a')


def test_success_is_cached_and_bound_to_the_exact_candidate(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);runner=fake_native(probe,monkeypatch)
    assert probe.probe('a')['policy_attack_success'] is True
    assert probe.probe('a')['cached'] is True
    runner.assert_called_once()
    assert probe.require_challenge('a')['candidate_id']=='a'
    probe.editor.nodes['a']['xy'][1,0]=2
    with pytest.raises(ValueError,match='different candidate'):probe.require_challenge('a')
    with pytest.raises(ValueError,match='different candidate'):probe.probe('a')


def test_preexisting_history_collision_cannot_be_attributed_to_vad(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);fake_native(probe,monkeypatch,history=True)
    assert probe.probe('a')['target_collision_detected'] is True
    assert probe.probe('a')['policy_attack_success'] is False
    with pytest.raises(ValueError,match='not hit'):probe.require_challenge('a')


def test_budget_is_shared_across_candidates_and_object_branches(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);runner=fake_native(probe,monkeypatch,collision=False)
    for key in ('a','b','c'):probe.probe(key)
    assert probe.probe('d')['status']=='budget_exhausted'
    assert runner.call_count==3
    branch=make_probe(tmp_path,monkeypatch,root=probe.root)
    assert branch.context()['remaining']==0
    assert branch.probe('a')['status']=='budget_exhausted'


def test_geometry_and_goal_must_pass_before_spending_native_budget(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);runner=fake_native(probe,monkeypatch)
    probe.editor.nodes['a']['result']['construction_goal']['goal_met']=False
    with pytest.raises(ValueError,match='trajectory and task'):probe.probe('a')
    assert probe.context()['calls']==0
    runner.assert_not_called()


def test_execution_failure_is_not_a_no_contact_result(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);runner=fake_native(probe,monkeypatch)
    runner.return_value=CompletedProcess([],1,'','failed')
    with pytest.raises(RuntimeError,match='execution failed'):probe.probe('a')
    assert 'a' not in probe.results
    assert json.loads((probe.root/'probe_001/error.json').read_text())['status']=='execution_failed'


def test_native_tool_eligibility_and_main_handoff(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch)
    from attack_tool_contract import trajectory_tools
    tool=dict(type='function',function=dict(name='probe_vadv2',parameters=dict(properties=dict(candidate_id=dict(type='string')))))
    available=trajectory_tools([tool],probe.editor.nodes,set(),{'a'},2)
    assert available[0]['function']['parameters']['properties']['candidate_id']['enum']==['b','c','d']
    assert trajectory_tools([tool],probe.editor.nodes,set(),set(),0)==[]
    from agent.context import summarize
    context=summarize('attack_vehicle_editor',dict(vadv2_probe_result_path='/probe.json',
        policy_collision_report=dict(target_collision_detected=True,target_track='target')))
    assert context['policy_collision_report']['target_collision_detected'] is True


def test_real_submit_method_enforces_native_probe_before_accepting(tmp_path,monkeypatch):
    import ast
    source=CORE/'attack_editor.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='AttackEditor')
    function=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='submit')
    function.decorator_list=[]
    writes=Mock()
    namespace=dict(atomic_json=writes)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    gate=Mock(side_effect=ValueError('untested native candidate'))
    editor=SimpleNamespace(submitted=None,inspected={'a'},nodes={'a':{'result':dict(reference_screen_passed=True)}},
        constraints=dict(vadv2_self_test=True),policy_probe=SimpleNamespace(require_challenge=gate),run_dir=tmp_path,
        public=lambda key:dict(candidate_id=key))
    with pytest.raises(ValueError,match='untested native'):namespace['submit'](editor,'a','candidate')
    assert editor.submitted is None
    writes.assert_not_called()
    gate.side_effect=None
    assert namespace['submit'](editor,'a','candidate')['status']=='accepted'
    assert editor.submitted=='a'
