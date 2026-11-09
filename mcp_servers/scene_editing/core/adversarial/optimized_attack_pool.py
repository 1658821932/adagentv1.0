"""Budgeted control optimization followed by full reference/behavior acceptance."""
import json

import numpy as np

from attack_objective import goal_cost
from experiment_runtime import atomic_json
from vehicle_control_search import ControlSearch


def construct_pool(editor, arguments):
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
    path = editor.run_dir / 'constructor-budget.json'
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
    atomic_json(path, budget)
    selected, verified = [], []
    completed = False

    def reserve(controls):
        budget['evaluations'] += 1
        atomic_json(path, budget)
        atomic_json(folder/'pending-evaluation.json', dict(
            attempt_id=f"attempt_{budget['evaluations']-before:03d}", controls=controls))

    def evaluate(controls, arrays, state, cost):
        record = dict(attempt_id=f"attempt_{budget['evaluations']-before:03d}",
            status='optimization_evaluated', model='bicycle.controls.v1', guidance_cost=cost,
            controls=controls, strict_screen_performed=False,
            scope='Continuous objective/road/obstacle guidance; not final acceptance')
        with (folder / 'attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
        (folder/'pending-evaluation.json').unlink()
        return record

    try:
        ranked = search.optimize(evaluate, remaining, reserve=reserve)
        paths = []
        # Strictly verify a few diverse finalists, including the unmodified parent
        # control baseline. Optimization guidance does not authorize submission.
        initial_record = next(record for _, _, record in ranked if record['attempt_id']=='attempt_001')
        proposals = [initial_record] + [record for _, _, record in ranked if record is not initial_record]
        for record in proposals:
            arrays, _ = search.simulate(record['controls'])
            xy = arrays[0]
            if any(np.sqrt(np.mean(np.sum((xy-previous)**2, axis=1))) < .10 for previous in paths):
                continue
            operation = dict(parent_candidate_id=arguments['parent_candidate_id'],
                controls=record['controls'], construction_goal=arguments, reason=arguments['reason'])
            candidate = editor.construct_controls(operation)
            category = ('goal_candidate' if candidate['reference_screen_passed'] and candidate['construction_goal']['goal_met']
                        else 'fallback' if candidate['reference_screen_passed'] else 'repair_material')
            verified.append(dict(category=category, attempt_id=record['attempt_id'], guidance_cost=record['guidance_cost'], **candidate))
            paths.append(xy)
            atomic_json(editor.run_dir/'candidates'/candidate['candidate_id']/'construction-source.json',
                        dict(pool_id=pool_id, attempt_id=record['attempt_id']))
            if len(verified) >= 6:
                break
        verified.sort(key=lambda candidate: (
            0 if candidate['category']=='goal_candidate' else 1 if candidate['category']=='fallback' else 2,
            len(candidate['failed_checks']), goal_cost(candidate['construction_goal']['dense']), candidate['guidance_cost']))
        selected = verified[:arguments['max_candidates']]
        completed = True
    finally:
        output = dict(pool_id=pool_id,
            status='interrupted' if not completed else 'constructed' if any(c['category']=='goal_candidate' for c in selected)
            else 'fallback' if any(c['category']=='fallback' for c in selected) else 'no_feasible_goal',
            candidates=selected, evaluations_spent=budget['evaluations']-before,
            run_evaluations_spent=budget['evaluations'], strict_finalist_count=len(verified),
            stop_reason='interrupted' if not completed else 'run_budget_exhausted' if budget['evaluations']>=MAX_INTERNAL_PER_RUN
            else 'call_budget_exhausted' if budget['evaluations']-before>=MAX_INTERNAL_PER_CALL else 'optimizer_converged',
            limits=dict(per_call=MAX_INTERNAL_PER_CALL, per_run=MAX_INTERNAL_PER_RUN),
            scope='Parent control sequence, continuous road/observed-obstacle guidance, full strict finalist checks')
        atomic_json(folder/'strict-finalists.json', verified)
        atomic_json(folder/'output.json', output)
    return output
