"""Stage-aware attack evidence and PDM tool exposure; no acceptance relaxation."""
from copy import deepcopy


def revision_evidence(constraints):
    evidence=(constraints.get('revision_feedback') or {}).get('evidence',{})
    if isinstance(evidence,str):
        return dict(result_path=evidence)
    return evidence if isinstance(evidence,dict) else {}


def goal_diagnostics(goal):
    failures, measurements = [], []
    if goal is None:
        return failures, measurements
    for source in ('dense','native_replay'):
        report = goal.get(source) or {}
        failures.extend(source+'.'+name for name in report.get('failed_checks',[]))
        for index, frame in enumerate(report.get('keyframes',[])):
            if not frame['passed']:
                measurements.append(dict(source=source,keyframe=index+1,
                    interval_s=frame['evaluated_interval_s'],actual=frame['actual'],
                    max_constraint_error=frame['max_constraint_error'],
                    requested=frame.get('requested'),actual_ranges=frame.get('actual_ranges')))
    return failures, measurements


def candidate_eligibility(result):
    geometry = bool(result['reference_screen_passed'])
    goal = result.get('construction_goal')
    task = (goal is None or bool(goal['goal_met'])) and not result.get('structured_objective_missing',False)
    failures,_ = goal_diagnostics(goal)
    if result.get('structured_objective_missing'):
        failures.append('declared_objective_missing')
    return dict(reference_geometry_passed=geometry,task_contract_passed=task,
                pdm_probe_eligible=geometry and task,task_failed_checks=failures)


def optimization_summary(goal):
    reports=[(source,(goal or {}).get(source,{}).get('preferences')) for source in ('dense','native_replay')]
    reports=[(source,report) for source,report in reports if isinstance(report,dict)]
    if not reports: return None
    missed=[source+'.'+check for source,report in reports for check in report.get('failed_checks',[])]
    example=None
    for source,report in reports:
        frame=next((frame for frame in report.get('keyframes',[]) if not frame['passed']),None)
        if frame is not None:
            example=dict(source=source,requested=frame.get('requested'),
                evaluated_interval_s=frame.get('evaluated_interval_s'),actual_ranges=frame.get('actual_ranges'))
            break
    return dict(met=all(report['goal_met'] for _,report in reports),missed_checks=missed,
        first_missed_interval=example,scope='optimization preference; assess original instruction separately')


def constructor_exhaustion(nodes, budget):
    """Stop bounded attack search only when no trajectory is ready for PDM/acceptance.

    A pending PDM check is not a failed construction. Existing valid candidates,
    including the checked seed, must retain the opportunity to be inspected/tested.
    """
    remaining=budget.get('remaining_evaluations')
    if remaining is None or remaining>0:
        return None
    failures=set()
    for node in nodes.values():
        result=node['result']
        eligibility=candidate_eligibility(result)
        if eligibility['pdm_probe_eligible']:
            return None
        failures.update(result.get('failed_checks',[]))
        failures.update(eligibility['task_failed_checks'])
    return dict(status='finished',stop_reason='constructor_budget_exhausted_without_qualified_candidate',
        reason='The cumulative attack construction evaluation budget is exhausted and no candidate passes both trajectory checks and the fixed task requirements.',
        constructor_budget=deepcopy(budget),checked_candidate_count=len(nodes),
        failed_checks=sorted(failures),scope='current_tools_and_budget',
        global_infeasibility_proven=False)


def trajectory_tools(tools, nodes, evaluated, tested, probes_remaining, *,
                     constructor_remaining=None, reselections_remaining=None, inspected=None):
    """Offer actions only while their eligibility and independent budgets permit them.

    Unknown balances preserve compatibility for callers without a constructor ledger.
    Exhausting construction does not consume point editing or repair actions.
    """
    available=[]
    for tool in deepcopy(tools):
        function=tool['function'];name=function['name']
        if name in ('edit_speed_profile','edit_route','repair_candidate') and any(
                'structured_objective_missing' in node['result'] for node in nodes.values()):
            ids=[key for key,node in nodes.items() if not node['result'].get('structured_objective_missing')]
            if not ids:
                continue
            function['parameters']['properties'].setdefault('parent_candidate_id',{'type':'string'})['enum']=ids
        if name=='construct_attack_candidates' and constructor_remaining is not None and constructor_remaining<=0:
            continue
        if name in ('request_object_reselection','request_ego_history_revision') and reselections_remaining is not None and reselections_remaining<=0:
            continue
        if name in ('probe_pdm','probe_vadv2'):
            ids=[key for key,node in nodes.items() if candidate_eligibility(node['result'])['pdm_probe_eligible'] and key not in tested]
            if probes_remaining<=0 or not ids:
                continue
            function['parameters']['properties']['candidate_id']['enum']=ids
        elif name=='evaluate_candidate':
            ids=list(nodes)
            if not ids:
                continue
            function['parameters']['properties']['candidate_id']['enum']=ids
        elif name=='inspect_trajectory' and inspected is not None:
            ids=list(nodes)
            if not ids:
                continue
            function['parameters']['properties']['candidate_id']['enum']=ids
        available.append(tool)
    return available
