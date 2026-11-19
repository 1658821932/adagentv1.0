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


def test_compaction_keeps_periods_for_all_and_curves_for_active_comparison_only():
    evidence=temporal_evidence([0.,3.,4.],[[0.,0.],[1.,0.],[1.,0.]],[5.,6.,7.],3.)
    result=compact_context(dict(candidate_order_note='generation order',selected_initial_candidate_id='z',candidate_catalog=[
        dict(candidate_id='z',status='reference_screen_passed',**evidence),
        dict(candidate_id='a',status='reference_screen_failed',failed_checks=['target_clearance'],**evidence)]))
    rows=[dict(zip(result['candidate_catalog']['columns'],r)) for r in result['candidate_catalog']['rows']]
    assert [r['candidate_id'] for r in rows]==['z','a']
    assert all('motion_timeline' not in r for r in rows)
    assert result['active_candidate_details']['z']['motion_timeline']==evidence['motion_timeline']
    assert all(r['clearance_periods']==evidence['clearance_periods'] for r in rows)
    assert 'numerical best' not in result['context_policy']


def test_generation_feedback_references_catalog_without_repeating_public_candidate():
    row=dict(candidate_id='new',status='reference_screen_failed',failed_checks=['road'],
             coverage=dict(road='failed',no_cross_lines='unknown'))
    result=compact_context(dict(candidate_catalog=[row],latest_generated_candidate_id='new',
        last_tool_result=dict(status='candidates_generated',candidates=[dict(row,speed_target_feedback=dict(targets_met=False))],failures=['optimizer timed out'])))
    assert result['last_tool_result']['candidates']==[dict(candidate_id='new',evidence='candidate_catalog',speed_target_feedback=dict(targets_met=False))]
    assert result['last_tool_result']['failures']==['optimizer timed out']
    table=result['candidate_catalog'];record=dict(zip(table['columns'],table['rows'][0]))
    assert record['failed_checks']==['road'] and record['unknown_checks']==['no_cross_lines']
    assert result['active_candidate_details']['new']['summary_evidence']=='candidate_catalog'


def test_percentage_display_does_not_remove_progress_from_legacy_non_behavior_runs():
    from mcp_servers.scene_editing.core.adversarial.ego_context_view import candidate_summary
    legacy=dict(candidate_id='old',progress_ratio_of_recorded=.6)
    assert candidate_summary(legacy)['progress_ratio_of_recorded']==.6
    modern=dict(legacy,behavior_metrics=dict(reference_progress_percent=47.5))
    assert 'progress_ratio_of_recorded' not in candidate_summary(modern)


def test_inspecting_another_candidate_expands_its_curves_without_changing_edit_parent():
    evidence=temporal_evidence([0.,3.,4.],[[0.,0.],[1.,0.],[1.,0.]],[5.,6.,7.],3.)
    rows=[dict(candidate_id=cid,status='reference_screen_passed',**evidence) for cid in ('fallback','another','unused')]
    context=dict(candidate_catalog=rows,current_edit_parent_id='fallback',inspection_candidate_ids=['another'])
    result=compact_context(context)
    assert set(result['active_candidate_details'])=={'fallback','another'}
    assert result['active_candidate_details']['another']['motion_timeline']==evidence['motion_timeline']
    assert result['current_edit_parent_id']=='fallback'


def test_after_selection_only_active_candidate_motion_is_sent():
    evidence=temporal_evidence([0.,3.,4.],[[0.,0.],[1.,0.],[1.,0.]],[5.,6.,7.],3.)
    rows=[dict(candidate_id=cid,status='reference_screen_passed',**evidence)
          for cid in ('fallback','inspected','unused','edit_001')]
    rows[2].update(status='reference_screen_failed',failed_checks=['road'],coverage={'rear':'unknown'})
    context=dict(candidate_catalog=rows,selected_initial_candidate_id='fallback',
        current_edit_parent_id='fallback',latest_generated_candidate_id='edit_001',
        inspection_candidate_ids=['inspected'])
    before=deepcopy(context)
    result=compact_context(context,focus_selected=True)
    table=result['candidate_catalog']
    catalog=[dict(zip(table['columns'],row)) for row in table['rows']]
    assert [row['candidate_id'] for row in catalog]==[row['candidate_id'] for row in rows]
    assert catalog[2]['reference_screen_passed'] is False
    assert catalog[2]['failed_checks']==['road'] and catalog[2]['unknown_checks']==['rear']
    assert 'clearance_periods' not in table['columns']
    active=result['active_candidate_details']
    assert set(active)=={'fallback','inspected','edit_001'}
    assert active['fallback']['motion_timeline']==evidence['motion_timeline']
    assert active['inspected']['clearance_periods']==evidence['clearance_periods']
    assert context==before
