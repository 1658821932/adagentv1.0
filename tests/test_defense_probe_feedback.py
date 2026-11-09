"""Defense feasibility advice remains separate from attack gates and API errors."""
import json
from pathlib import Path

import pytest
from mcp_servers.scene_editing.core.adversarial import defense_probe_feedback as feedback
from scripts.run_adversarial_validation import defense_revision_feedback


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def summary(passed=True):
    return dict(status='reference_screen_complete',qualified_count=int(passed),construction_failures=[],
        candidates=[dict(candidate_id='candidate_001',reference_screen_passed=passed,
            shared_checks=dict(target_clearance=passed,acceleration=True),
            min_target_clearance_m=.2 if passed else 0.,max_acceleration_mps2=5.)],
        defense_difficulty=dict(edit_start_clearance_m=.35,straight_line_stop_estimate=dict(distance_m=1.53)))


@pytest.mark.parametrize('passed',[False,True])
def test_advice_never_claims_global_infeasibility_or_changes_attack_acceptance(tmp_path,passed):
    result=feedback.summarize(summary(passed),tmp_path)
    assert result['qualified_count']==int(passed)
    assert result['outcome']==('reference_witness_found' if passed else 'no_reference_witness_found')
    assert result['diagnostic_only'] and not result['attack_acceptance_affected']
    assert not result['global_feasibility_proven']
    assert result['verified_passing_candidate_ids']==(['candidate_001'] if passed else [])
    assert result['failed_check_counts']==({} if passed else dict(target_clearance=1))


def test_inconsistent_constructor_summary_rejected(tmp_path):
    value=summary();value['qualified_count']=0
    with pytest.raises(ValueError,match='contradicts'):feedback.summarize(value,tmp_path)


@pytest.fixture
def pool(tmp_path):
    scene=tmp_path/'scene';write(scene/'metadata/frames.json',[dict(timestamp=1,ego2global=[[1]],anns={})])
    write(scene/'metadata/map_drivable_official.json',dict(features={}))
    acceptance=dict(profile='test',allow_safe_stop=True)
    source=tmp_path/'constructor';write(source/'reference_screen/summary.json',summary())
    write(source/'probe-binding.json',feedback.constructor_binding(scene,'target',acceptance,24))
    probe=tmp_path/'probe.json';write(probe,dict(defense_feasibility=dict(status='completed',
        constructor_run=str(source),qualified_count=1)))
    return scene,acceptance,source,probe


def test_compatible_pool_reused_with_explicit_nonindependent_provenance(pool,tmp_path):
    scene,acceptance,source,probe=pool;output=tmp_path/'defense_pool'
    assert feedback.reuse(probe,scene,'target',acceptance,24,output)
    value=json.loads((output/'reference_screen/summary.json').read_text())
    assert value['pool_provenance']['contains_validated_witness']
    assert not value['pool_provenance']['agent_independent_rediscovery_established']


@pytest.mark.parametrize('change',['frames','map','policy','target','budget'])
def test_changed_scene_policy_or_budget_recomputes_instead_of_reusing(pool,tmp_path,change):
    scene,acceptance,source,probe=pool;count=24;target='target'
    if change=='frames':write(scene/'metadata/frames.json',[dict(timestamp=2)])
    elif change=='map':write(scene/'metadata/map_drivable_official.json',dict(features={'new':{}}))
    elif change=='policy':acceptance=dict(acceptance,allow_safe_stop=False)
    elif change=='target':target='other'
    else:count=25
    output=tmp_path/'defense_pool'
    assert not feedback.reuse(probe,scene,target,acceptance,count,output)
    assert not output.exists()


def test_unavailable_advice_is_never_a_zero_candidate_result(tmp_path):
    probe=tmp_path/'probe.json';write(probe,dict(defense_feasibility=dict(status='unavailable',error_type='RuntimeError')))
    assert not feedback.reuse(probe,tmp_path,'target',{},24,tmp_path/'output')


@pytest.mark.parametrize('status,reason',[('backend_error',None),('defense_not_found','request_or_execution_failed')])
def test_execution_error_does_not_reduce_attack_strength(status,reason):
    response=dict(status=status,detail=dict(status='not_submitted',stop_reason=reason))
    assert defense_revision_feedback(response,{}) is None


def test_completed_bounded_search_can_request_less_demanding_attack():
    response=dict(status='defense_not_found',detail=dict(status='not_submitted',stop_reason='model_turn_budget_exhausted',
        model_actions=10,edit_candidate_count=5,best_min_target_clearance_m=0.))
    attack=dict(status='success',tool='attack_vehicle_editor',result=dict(selected_candidate_id='seed_000',
        submission_path='/run/trajectory_editing/output.json',target_track='target'))
    result=defense_revision_feedback(response,attack)
    assert result['stage']=='defense' and not result['evidence']['global_feasibility_proven']


@pytest.mark.parametrize('curve_peak,expected', [(5.609, 5.609), (4.0, 5.0), (None, 5.0)])
def test_witness_reports_native_and_curve_acceleration_separately(tmp_path, curve_peak, expected):
    value = summary()
    if curve_peak is not None:
        value['candidates'][0]['generation_curve'] = dict(max_acceleration_mps2=curve_peak)
    witness = feedback.summarize(value, tmp_path)['witnesses'][0]
    assert witness['native_max_acceleration_mps2'] == 5.0
    assert witness['generated_curve_max_acceleration_mps2'] == curve_peak
    assert witness['max_acceleration_mps2'] == expected
