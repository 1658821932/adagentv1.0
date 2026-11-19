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
    monkeypatch.setenv('ADVERSARIAL_HOST_HOOKS', 'mcp_servers.scene_editing.adapter.core_hooks')
    from attack_policy_probe import VADProbe
    editor=SimpleNamespace(constraints=dict(vadv2_probe_root=str(root or tmp_path/'probes'),
        vadv2_probe_budget=3,policy_source_scene_ref={},vadv2_policy_config=dict(python=sys.executable,
            worldengine_root=str(tmp_path),config_path='/cfg',checkpoint_path='/weights')),
        scene=tmp_path/'scene',run_dir=tmp_path/'editor',times=np.arange(24)*.5,
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
    with pytest.raises(ValueError,match='no confirmed'):probe.require_challenge('a')


@pytest.mark.parametrize(('lateral_m', 'map_radius_m', 'road_status', 'success'), [
    (0., 100, 'passed', False),
    (6., 100, 'failed', True),
    (106., 100, 'inconclusive', False),
])
def test_no_collision_uses_actual_road_compliance(
        tmp_path, monkeypatch, lateral_m, map_radius_m, road_status, success):
    from mcp_servers.scene_editing.policy_evidence import has_policy_failure

    probe = make_probe(tmp_path, monkeypatch)
    runner = fake_native(probe, monkeypatch, collision=False)
    probe.editor.frames = [dict(ego2global=np.eye(4))]
    archived = tmp_path/'states.jsonl'
    rows = [
        dict(step=i, time_s=i/2, actors=dict(ego=dict(
            box=[0., lateral_m if i >= 8 else 0., 4., 2., 0.], velocity=[0., 0.])))
        for i in range(16)
    ]
    archived.write_text('\n'.join(map(json.dumps, rows)))
    response = json.loads(runner.return_value.stdout)
    response['result']['collision_report']['states_path'] = str(archived)
    runner.return_value.stdout = json.dumps(response)

    original_materialize = probe._materialize
    def materialize(candidate, folder):
        frozen = original_materialize(candidate, folder)
        (frozen/'metadata/map_drivable_official.json').write_text(json.dumps(dict(
            map_radius_m=map_radius_m,
            features={'road': {'polygon': [[-20, -3], [20, -3], [20, 3], [-20, 3]]}})))
        return frozen
    probe._materialize = materialize
    result = probe.probe('a')
    evaluation = result['policy_evaluation']
    assert result['target_collision_detected'] is False
    assert result['policy_attack_success'] is success
    assert evaluation['drivable_area']['status'] == road_status
    if success:
        assert evaluation['failure_reasons'] == ['drivable_area_violation']
        assert evaluation['drivable_area']['first_violation_time_s'] == 4.
        assert probe.require_challenge('a')['candidate_id'] == 'a'
        evidence = json.loads(Path(result['result_path']).read_text())
        road_map = Path(evidence['frozen_scene_path'])/'metadata/map_drivable_official.json'
        assert has_policy_failure(evidence, road_map)
        road_map.write_text('{}')
        assert not has_policy_failure(evidence, road_map)


def test_success_is_cached_and_bound_to_the_exact_candidate(tmp_path,monkeypatch):
    from attack_tool_contract import submission_ready_candidates

    probe=make_probe(tmp_path,monkeypatch);runner=fake_native(probe,monkeypatch)
    editor=probe.editor
    editor.constraints['vadv2_self_test']=True
    editor.policy_probe=probe
    editor.pdm_probe=None
    assert probe.probe('a')['policy_attack_success'] is True
    assert probe.probe('a')['cached'] is True
    runner.assert_called_once()
    assert probe.require_challenge('a')['candidate_id']=='a'
    assert submission_ready_candidates(editor,['a','b'],{'b'})==[]
    assert submission_ready_candidates(editor,['a','b'],{'a','b'})==['a']
    editor.nodes['a']['xy'][1,0]=2
    assert submission_ready_candidates(editor,['a','b'],{'a','b'})==[]
    with pytest.raises(ValueError,match='different candidate'):probe.require_challenge('a')
    with pytest.raises(ValueError,match='different candidate'):probe.probe('a')


def test_long_scene_probe_preserves_standard_window_and_native_inputs(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);runner=fake_native(probe,monkeypatch)
    result=probe.probe('a')
    request=json.loads(runner.call_args.kwargs['input'])
    assert (request['num_history'],request['num_future'],request['max_frame_per_scene'])==(4,12,16)
    assert request['collision_only'] is True
    assert request['visualize_video'] is request['visualize_BEV'] is request['with_dense_reward_manager'] is False
    assert request['react_type']=='NR' and request['collision_target_track']=='target'
    assert request['config_path']=='/cfg' and request['checkpoint_path']=='/weights'
    assert result['evaluation_window_s']==[1.5,7.5]


def test_short_source_does_not_spend_probe_budget_or_shorten_window(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);runner=fake_native(probe,monkeypatch)
    probe.editor.times=np.arange(12)*.5
    with pytest.raises(ValueError,match='16 source frames'):probe.probe('a')
    assert probe.context()['calls']==0
    runner.assert_not_called()


@pytest.mark.parametrize('collision_time',[1.,1.5,8.])
def test_contact_outside_controlled_window_is_not_attack_success(tmp_path,monkeypatch,collision_time):
    probe=make_probe(tmp_path,monkeypatch);runner=fake_native(probe,monkeypatch)
    payload=json.loads(runner.return_value.stdout)
    payload['result']['collision_report']['first_target_collision_time_s']=collision_time
    runner.return_value.stdout=json.dumps(payload)
    assert probe.probe('a')['policy_attack_success'] is False


def test_feedback_is_compact_but_keeps_avoidance_evidence(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);runner=fake_native(probe,monkeypatch,collision=False)
    payload=json.loads(runner.return_value.stdout)
    payload['result']['policy_behavior_summary']=dict(status='completed',stopped_detected=True,
        stopped_from_time_s=5.,progress_ratio_vs_recorded=.27,signed_yaw_change_deg=-12.,
        scenes=[{'verbose':'archive only'}],pdm_subscores={'score':1})
    runner.return_value.stdout=json.dumps(payload)
    result=probe.probe('a')
    assert result['policy_behavior_summary']==dict(status='completed',stopped_detected=True,
        stopped_from_time_s=5.,progress_ratio_vs_recorded=.27,signed_yaw_change_deg=-12.)
    assert 'pdm_subscores' not in result


def test_preexisting_history_collision_cannot_be_attributed_to_vad(tmp_path,monkeypatch):
    probe=make_probe(tmp_path,monkeypatch);fake_native(probe,monkeypatch,history=True)
    assert probe.probe('a')['target_collision_detected'] is True
    assert probe.probe('a')['policy_attack_success'] is False
    with pytest.raises(ValueError,match='no confirmed'):probe.require_challenge('a')


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
    monkeypatch.syspath_prepend(str(CORE))
    import attack_editor

    writes=Mock()
    monkeypatch.setattr(attack_editor,'atomic_json',writes)
    submit=attack_editor.AttackEditor.submit.__wrapped__
    gate=Mock(side_effect=ValueError('untested native candidate'))
    editor=SimpleNamespace(submitted=None,inspected={'a'},nodes={'a':{'result':dict(reference_screen_passed=True)}},
        constraints=dict(vadv2_self_test=True),pdm_probe=None,policy_probe=SimpleNamespace(require_challenge=gate),run_dir=tmp_path,
        public=lambda key:dict(candidate_id=key))
    with pytest.raises(ValueError,match='untested native'):submit(editor,'a','candidate')
    assert editor.submitted is None
    writes.assert_not_called()
    gate.side_effect=None
    assert submit(editor,'a','candidate')['status']=='accepted'
    assert editor.submitted=='a'
