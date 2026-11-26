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
        for name in ('waypoints', 'requested_waypoints'):
            for index, waypoint in enumerate(report.get(name, []), 1):
                if not waypoint['passed']:
                    measurements.append(dict(source=source, waypoint=index, waypoint_group=name,
                        time_s=waypoint['evaluated_time_s'], requested=waypoint['requested'],
                        actual=waypoint['actual'], errors=waypoint['errors'], tolerances=waypoint['tolerances']))
    return failures, measurements


def _goal_is_soft_preference(goal, constraints=None):
    if goal is None:
        return True
    if constraints and (constraints.get('task_contract') or constraints.get('objective_contract')):
        return False
    return bool(constraints and constraints.get('generated_objectives_are_preferences')) or (
        isinstance(goal.get('keyframe_scope'), str) and 'preferences' in goal.get('keyframe_scope', '').lower()
    )


def candidate_eligibility(result):
    geometry = bool(result['reference_screen_passed'])
    goal = result.get('construction_goal')
    constraints = result.get('effective_constraints') or {}
    goal_ok = goal is None or bool(goal['goal_met']) or _goal_is_soft_preference(goal, constraints)
    task = goal_ok and not result.get('structured_objective_missing',False)
    failures,_ = goal_diagnostics(goal)
    if result.get('structured_objective_missing'):
        failures.append('declared_objective_missing')
    return dict(reference_geometry_passed=geometry,task_contract_passed=task,
                pdm_probe_eligible=geometry and task,task_failed_checks=failures)


def check_submission(editor, candidate_id):
    """Validate the candidate and its exact-version policy evidence."""
    result = editor.nodes[candidate_id]['result']
    if not result['reference_screen_passed']:
        raise ValueError('Submitted candidate must pass all configured reference checks')
    declared = result.get('construction_goal_input', {})
    constraints = editor.constraints
    if (constraints.get('task_contract') or (constraints.get('require_structured_objective')
            and not constraints.get('stationary_target'))) and 'objective' not in declared:
        raise ValueError('Declare and pass a structured objective before submitting this run')
    if constraints.get('behavior_contract') and declared.get('interaction_mode') == 'either':
        raise ValueError('Declare near_miss/contact through active construction before submission')
    goal = result.get('construction_goal')
    if goal is not None and not goal['goal_met'] and not _goal_is_soft_preference(goal, constraints):
        raise ValueError('Active construction goal is not met; fallback cannot be submitted as goal completion')
    for setting, probe, name in (
        ('pdm_self_check', editor.pdm_probe, 'PDM'),
        ('vadv2_self_test', editor.policy_probe, 'VADv2'),
    ):
        if constraints.get(setting):
            if probe is None:
                raise ValueError(name + ' self-check was enabled but its controller is unavailable')
            probe.require_challenge(candidate_id)


def submission_ready_candidates(editor, qualified, reviewed):
    ready = []
    for candidate_id in qualified:
        if candidate_id not in reviewed:
            continue
        try:
            check_submission(editor, candidate_id)
        except ValueError:
            continue
        ready.append(candidate_id)
    return ready


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
