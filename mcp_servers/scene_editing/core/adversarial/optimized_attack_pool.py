"""Budgeted control optimization followed by full reference/behavior acceptance."""
import json

import numpy as np

from attack_objective import goal_cost
from experiment_runtime import atomic_json
from attack_budget import constructor_budget_path
from vehicle_control_search import ControlSearch


MIN_STRICT_FINALISTS=6


def finalist_order(ranked):
    ordered=[]
    seen=set()
    def add(item):
        identifier=item[2]['attempt_id']
        if identifier not in seen:
            seen.add(identifier);ordered.append(item)
    initial=next(item for item in ranked if item[2]['attempt_id']=='attempt_001')
    add(initial)
    feasible=[item for item in ranked if item[2].get('search_metrics',{}).get('geometry_motion_guidance_feasible')]
    for item in sorted(feasible, key=lambda item:item[0])[:MIN_STRICT_FINALISTS]:
        add(item)
    contact=[item for item in ranked if item[2].get('search_metrics',{}).get('window_reference_contact')]
    if contact:add(min(contact,key=lambda item:item[0]))
    timely=[item for item in ranked if not item[2].get('search_metrics',{}).get('late_contact_seconds',0.)]
    if timely:
        add(min(timely,key=lambda item:(
            item[2].get('search_metrics',{}).get('motion_guidance_violation_count',0),
            max(0.,item[2].get('search_metrics',{}).get('window_min_signed_gap_m',0.)),item[0])))
    for item in ranked:add(item)
    return ordered


def construct_pool(editor, arguments, connector_plan=None):
    from attack_editor import MAX_INTERNAL_PER_CALL, MAX_INTERNAL_PER_RUN
    if editor.constraints.get('behavior_contract'):
        raise ValueError('This archived run has a legacy behavior contract; use a new run for structured objectives')
    contract = editor.constraints.get('objective_contract')
    if contract is not None and arguments['objective'] != contract:
        raise ValueError('Cannot alter the configured objective contract')
    event = editor.constraints.get('event_window_s')
    window = arguments['objective']['interaction_window_s']
    if event is not None and (window[0] < event[0] or window[1] > event[1]):
        raise ValueError('Interaction window must stay inside configured event window')
    search = ControlSearch(editor, arguments)
    path = constructor_budget_path(editor.run_dir)
    budget = json.loads(path.read_text()) if path.exists() else dict(evaluations=0, calls=0)
    remaining = min(MAX_INTERNAL_PER_CALL, MAX_INTERNAL_PER_RUN-budget['evaluations'])
    if remaining <= 0:
        raise ValueError('Active constructor run evaluation budget exhausted')
    before = budget['evaluations']
    budget['calls'] += 1
    pool_id = f"pool_{budget['calls']:03d}"
    folder = editor.run_dir / 'construction_pools' / pool_id
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / 'input.json', arguments)
    if connector_plan is not None:
        atomic_json(folder / 'connector-route.json', connector_plan)
    atomic_json(path, budget)
    selected, verified = [], []
    completed = False
    strict_stop_reason='not_started'

    def reserve(controls):
        budget['evaluations'] += 1
        atomic_json(path, budget)
        atomic_json(folder/'pending-evaluation.json', dict(
            attempt_id=f"attempt_{budget['evaluations']-before:03d}", controls=controls,
            initial_pose_adjustment=search.active_pose))

    def evaluate(controls, arrays, state, cost):
        record = dict(attempt_id=f"attempt_{budget['evaluations']-before:03d}",
            status='optimization_evaluated', model='bicycle.controls.v1', guidance_cost=cost,
            controls=controls, strict_screen_performed=False,
            search_metrics=dict(search.last_metrics), search_phase=search.search_phase,
            initial_pose_adjustment=search.active_pose,
            scope='Continuous objective/road/obstacle guidance; not final acceptance')
        if search.continuation_source is not None:
            record['continuation_source'] = search.continuation_source
        with (folder / 'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        (folder/'pending-evaluation.json').unlink()
        return record

    try:
        ranked = search.optimize(evaluate, remaining, reserve=reserve)
        paths = []
        records={record['attempt_id']:record for _,_,record in ranked}
        def operation(record):
            return dict(parent_candidate_id=arguments['parent_candidate_id'],
                controls=record['controls'], construction_goal=arguments, reason=arguments['reason'],
                initial_pose_adjustment=record.get('initial_pose_adjustment'))
        initial_record = records['attempt_001']
        proposals = [initial_record] + [record for _, _, record in finalist_order(ranked) if record is not initial_record]
        strict_stop_reason='all_optimization_records_screened'
        for record in proposals:
            arrays, _ = search.simulate(record['controls'],record.get('initial_pose_adjustment'))
            xy = arrays[0]
            if any(np.sqrt(np.mean(np.sum((xy-previous)**2, axis=1))) < .10 for previous in paths):
                continue
            candidate = editor.construct_controls(operation(record),persist=False)
            candidate['candidate_id']=None
            category = ('goal_candidate' if candidate['reference_screen_passed'] and candidate['construction_goal']['goal_met']
                        else 'fallback' if candidate['reference_screen_passed'] else 'repair_material')
            verified.append(dict(category=category, attempt_id=record['attempt_id'], guidance_cost=record['guidance_cost'],
                search_metrics=record.get('search_metrics',{}), search_phase=record.get('search_phase'),
                continuation_source=record.get('continuation_source'), **candidate))
            paths.append(xy)
            if len(verified)>=MIN_STRICT_FINALISTS and any(item.get('pdm_probe_eligible') for item in verified):
                strict_stop_reason='qualified_candidate_found_after_minimum'
                break
        verified.sort(key=lambda candidate: (
            0 if candidate['category']=='goal_candidate' else 1 if candidate['category']=='fallback' else 2,
            len(candidate['failed_checks']), goal_cost(candidate['construction_goal']['dense']), candidate['guidance_cost']))
        for entry in verified[:arguments['max_candidates']]:
            record=records[entry['attempt_id']]
            candidate=editor.construct_controls(operation(record),persist=True)
            category=('goal_candidate' if candidate['reference_screen_passed'] and candidate['construction_goal']['goal_met']
                else 'fallback' if candidate['reference_screen_passed'] else 'repair_material')
            persisted=dict(entry,**candidate);persisted['category']=category
            selected.append(persisted)
            entry['selected_candidate_id']=candidate['candidate_id']
            atomic_json(editor.run_dir/'candidates'/candidate['candidate_id']/'construction-source.json',
                dict(pool_id=pool_id, attempt_id=record['attempt_id']))
        completed = True
    finally:
        output = dict(pool_id=pool_id,
            status='interrupted' if not completed else 'constructed' if any(c['category']=='goal_candidate' for c in selected)
            else 'fallback' if any(c['category']=='fallback' for c in selected) else 'no_feasible_goal',
            candidates=selected, evaluations_spent=budget['evaluations']-before,
            rejected_initial_poses=search.rejected_initial_poses,
            run_evaluations_spent=budget['evaluations'], strict_finalist_count=len(verified),
            strict_stop_reason=strict_stop_reason,minimum_strict_finalists=MIN_STRICT_FINALISTS,
            stop_reason='interrupted' if not completed else 'run_budget_exhausted' if budget['evaluations']>=MAX_INTERNAL_PER_RUN
            else 'call_budget_exhausted' if budget['evaluations']-before>=MAX_INTERNAL_PER_CALL else 'optimizer_converged',
            limits=dict(per_call=MAX_INTERNAL_PER_CALL, per_run=MAX_INTERNAL_PER_RUN),
            scope='Parent control sequence, continuous guidance and adaptive strict screening; only selected results become formal candidates')
        if connector_plan is not None:
            from connector_route import connector_passage
            output['connector_route'] = {key: connector_plan[key] for key in ('connector_id', 'entry_lane', 'exit_lane')}
            for candidate in selected:
                node = editor.nodes[candidate['candidate_id']]
                candidate['connector_passage'] = connector_passage(editor.scene, connector_plan, editor.times, node['xy'])
        atomic_json(folder/'strict-finalists.json', verified)
        atomic_json(folder/'output.json', output)
    return output
