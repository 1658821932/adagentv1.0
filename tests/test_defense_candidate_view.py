"""Candidate choice uses unranked temporal evidence, preserving eligibility and history."""
from copy import deepcopy
from mcp_servers.scene_editing.core.adversarial.defense_candidate_view import temporal_evidence, unranked_view
from mcp_servers.scene_editing.core.adversarial.ego_context_view import compact_context


def test_fixed_history_minimum_is_separated_from_editable_and_reference_contact_period():
    times=[0.,1.,2.,3.,4.,5.,6.]
    xy=[[float(i),0.] for i in range(7)]
    gaps=[29.58,30.8,31.,30.55,29.94,30.6,32.7]
    evidence=temporal_evidence(times,xy,gaps,3.,[5,6])
    periods=evidence['clearance_periods']
    assert periods['observation']['minimum_time_s']==0.
    assert periods['editable']['minimum_time_s']==4.
    assert periods['editable']['min_clearance_m']==29.94
    assert periods['recorded_ego_contact_samples']['min_clearance_m']==30.6
    assert evidence['motion_timeline']['target_clearance_m']==gaps
    assert evidence['motion_outcome']['editable_travel_m']==3.
    assert not evidence['motion_outcome']['terminal_stopped']
    assert evidence['motion_outcome']['terminal_stop_first_sample_s'] is None


def test_stop_timing_and_empty_contact_diagnostics_do_not_invent_a_window():
    evidence=temporal_evidence([0.,2.,3.,4.,5.],[[0.,0.],[2.,0.],[3.,0.],[3.,0.],[3.,0.]],
        [10.,9.,8.,8.,8.],3.)
    assert evidence['clearance_periods']['recorded_ego_contact_samples'] is None
    assert evidence['motion_outcome']['terminal_stop_first_sample_s']==3.
    assert evidence['motion_outcome']['terminal_stopped']
    assert evidence['motion_outcome']['editable_travel_m']==0.


def test_recursive_rank_removal_preserves_candidate_order_checks_and_explicit_selection():
    context=dict(pool_preferred_candidate_id='b',best={'candidate_id':'b'},selected_candidate_id='a',
        candidate_catalog=[dict(candidate_id='b',failed_checks=['road']),dict(candidate_id='a',failed_checks=[])],
        last_tool_result=dict(candidates=[dict(candidate=dict(candidate_id='c'),recommended_next_parent_id='b',
            is_new_best=True,productive_edit=True)],new_constraint_failures=['motion']))
    before=deepcopy(context);view=unranked_view(context)
    assert context==before
    assert 'best' not in view and 'pool_preferred_candidate_id' not in view
    assert view['selected_candidate_id']=='a'
    assert [row['candidate_id'] for row in view['candidate_catalog']]==['b','a']
    assert view['candidate_catalog'][0]['failed_checks']==['road']
    assert view['last_tool_result']['candidates']==[dict(candidate=dict(candidate_id='c'))]
    assert view['last_tool_result']['new_constraint_failures']==['motion']


def test_compaction_keeps_temporal_curves_for_every_candidate_without_recommendations():
    evidence=temporal_evidence([0.,3.,4.],[[0.,0.],[1.,0.],[1.,0.]],[5.,6.,7.],3.)
    result=compact_context(dict(candidate_order_note='generation order',candidate_catalog=[
        dict(candidate_id='z',status='reference_screen_passed',**evidence),
        dict(candidate_id='a',status='reference_screen_failed',failed_checks=['target_clearance'],**evidence)]))
    rows=[dict(zip(result['candidate_catalog']['columns'],r)) for r in result['candidate_catalog']['rows']]
    assert [r['candidate_id'] for r in rows]==['z','a']
    assert all(r['motion_timeline']==evidence['motion_timeline'] for r in rows)
    assert all(r['clearance_periods']==evidence['clearance_periods'] for r in rows)
    assert 'numerical best' not in result['context_policy']
