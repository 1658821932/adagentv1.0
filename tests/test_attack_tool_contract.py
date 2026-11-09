"""Native replay goal failures are visible and ineligible actions are not offered."""
from copy import deepcopy
from mcp_servers.scene_editing.core.adversarial.attack_tool_contract import candidate_eligibility,constructor_exhaustion,goal_diagnostics,trajectory_tools


def result(passed):
    frame=dict(passed=passed,evaluated_interval_s=[9.,9.5],actual={},max_constraint_error=.21,
               requested={'speed_mps':[0.,5.]},actual_ranges={'speed_mps':[4.7,5.21]})
    return dict(reference_screen_passed=True,construction_goal=dict(goal_met=passed,
        dense=dict(failed_checks=[],keyframes=[]),
        native_replay=dict(failed_checks=[] if passed else ['keyframe_1'],keyframes=[frame])))


def tool(name):
    return dict(type='function',function=dict(name=name,parameters=dict(properties=dict(candidate_id=dict(type='string')))))


def test_native_failure_is_reported_despite_geometry_pass():
    evidence=candidate_eligibility(result(False))
    assert evidence['reference_geometry_passed'] and not evidence['task_contract_passed']
    assert not evidence['pdm_probe_eligible']
    assert evidence['task_failed_checks']==['native_replay.keyframe_1']
    _,reports=goal_diagnostics(result(False)['construction_goal'])
    assert reports[0]['source']=='native_replay' and reports[0]['actual_ranges']['speed_mps'][1]==5.21


def test_probe_enum_only_contains_ready_untested_candidates_and_read_is_not_repeated():
    tools=[tool('probe_pdm'),tool('evaluate_candidate'),tool('edit_trajectory_points')];before=deepcopy(tools)
    nodes={key:dict(result=result(passed)) for key,passed in [('bad',False),('tested',True),('ready',True)]}
    exposed=trajectory_tools(tools,nodes,{'bad'},{'tested'},1)
    byname={entry['function']['name']:entry['function'] for entry in exposed}
    assert byname['probe_pdm']['parameters']['properties']['candidate_id']['enum']==['ready']
    assert 'bad' not in byname['evaluate_candidate']['parameters']['properties']['candidate_id']['enum']
    assert tools==before


def test_no_eligible_candidate_hides_probe_without_loosening_the_gate():
    exposed=trajectory_tools([tool('probe_pdm'),tool('evaluate_candidate')],{'bad':dict(result=result(False))},{'bad'},set(),2)
    assert exposed==[]


def test_exhausted_budgets_hide_only_the_affected_tools():
    tools=[tool(name) for name in ('construct_attack_candidates','request_object_reselection',
        'edit_trajectory_points','repair_candidate','inspect_trajectory','finish_search')]
    before=deepcopy(tools)
    exposed=trajectory_tools(tools,{},set(),set(),0,
        constructor_remaining=0,reselections_remaining=0)
    assert [entry['function']['name'] for entry in exposed]==[
        'edit_trajectory_points','repair_candidate','inspect_trajectory','finish_search']
    assert tools==before
    # The budgets are independent; a new object must not reset inherited construction usage.
    exposed=trajectory_tools(tools,{},set(),set(),0,
        constructor_remaining=0,reselections_remaining=1)
    assert 'request_object_reselection' in {entry['function']['name'] for entry in exposed}
    exposed=trajectory_tools(tools,{},set(),set(),0,
        constructor_remaining=1,reselections_remaining=0)
    assert 'construct_attack_candidates' in {entry['function']['name'] for entry in exposed}


def test_constructor_exhaustion_requires_cumulative_budget_and_no_qualified_candidate():
    budget=dict(evaluations=1000,calls=5,remaining_evaluations=0)
    nodes={'road_bad':dict(result=dict(reference_screen_passed=False,failed_checks=['road_surface'])),
           'task_bad':dict(result=result(False))}
    report=constructor_exhaustion(nodes,budget)
    assert report['stop_reason']=='constructor_budget_exhausted_without_qualified_candidate'
    assert report['failed_checks']==['native_replay.keyframe_1','road_surface']
    assert not report['global_infeasibility_proven']
    assert constructor_exhaustion(nodes,dict(budget,remaining_evaluations=1)) is None
    assert constructor_exhaustion(nodes,dict(calls=0,evaluations=0)) is None
    # A valid trajectory must still get its pending PDM/inspection opportunity.
    nodes['ready']=dict(result=result(True))
    assert constructor_exhaustion(nodes,budget) is None
    budget['calls']=6
    assert report['constructor_budget']['calls']==5
