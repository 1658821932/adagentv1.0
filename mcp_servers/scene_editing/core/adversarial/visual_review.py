"""Visual eligibility and tool availability share one explicit, recoverable ledger."""
from copy import deepcopy


def review_state(nodes, inspections, deliveries, attached, pending_review, manual_limit):
    delivered = sorted({row['candidate_id'] for row in deliveries})
    passed = sorted(key for key, node in nodes.items() if node['result']['status'] == 'reference_screen_passed')
    attached_id = None if attached is None else attached['candidate_id']
    attached_ids = [] if attached is None else [row['candidate_id'] for row in attached.get('batch',[attached])]
    available_images = set(delivered) | set(attached_ids)
    regular = sum(not row.get('automatic', False) and not row.get('closing_review', False) for row in inspections)
    closing = sum(row.get('closing_review', False) for row in inspections)
    return dict(delivered_candidate_ids=delivered, reference_passed_candidate_ids=passed,
        submittable_candidate_ids=sorted(set(passed) & available_images),
        pending_review_candidate_id=pending_review, attached_candidate_id=attached_id,
        attached_candidate_ids=attached_ids,
        cached_inspection_candidate_ids=sorted({row['candidate_id'] for row in inspections if row.get('candidate_id')}),
        manual_inspections_remaining=max(0, manual_limit-regular), final_review_remaining=max(0, 1-closing),
        eligibility_scope='Attached image becomes recorded as delivered only when this request completes successfully; tool execution follows that recording.')


def tool_contract(base_tools, inspect_tool, state, *, selected, first_inspection, closing, final,
                  edit_names, require_edit=False, edit_count=0):
    tools = deepcopy(base_tools)
    ready = state['submittable_candidate_ids'] if not require_edit or edit_count else []
    passed = state['reference_passed_candidate_ids'] if not require_edit or edit_count else []
    if closing:
        tools = [t for t in tools if t['function']['name'] not in set(edit_names) | {'repair_candidate'}]
    tools = [t for t in tools if t['function']['name'] != 'submit_candidate' or (ready if final else passed)]
    for tool in tools:
        if tool['function']['name'] == 'submit_candidate':
            tool['function']['description'] = ('Submit a reference-passed candidate whose own image has been delivered. '
                'Read visual_submission_state. Unseen candidates are explicitly rejected with a required inspection ID.')
            tool['function']['parameters']['properties']['candidate_id']['enum'] = ready if final else passed
    choice = 'auto'
    if selected is None:
        return {'type':'function','function':{'name':'select_constructor_candidate'}}, tools
    if final:
        name = 'submit_candidate' if ready else 'finish_search'
        tools = [t for t in tools if t['function']['name'] == name]
        return {'type':'function','function':{'name':name}}, tools
    can_inspect = state['manual_inspections_remaining'] > 0 or (closing and state['final_review_remaining'] > 0)
    cached = state.get('cached_inspection_candidate_ids', [])
    if can_inspect or cached:
        item = deepcopy(inspect_tool)
        if not can_inspect:
            item['function']['parameters']['properties']['candidate_id']['enum'] = cached
        if first_inspection:
            item['function']['parameters']['properties']['candidate_id']['enum'] = [selected]
            choice = {'type':'function','function':{'name':'inspect_trajectory'}}
        tools.append(item)
    return choice, tools


def finalization_tools(tools, state, feedback, *, refinement_pending):
    """Close a checked created candidate without bypassing refinement or image delivery."""
    summary = (feedback or {}).get('decision_summary') or feedback or {}
    candidate_id = summary.get('candidate_id')
    if (refinement_pending or summary.get('outcome') != 'candidate_checks_passed'
            or not summary.get('candidate_created')
            or candidate_id not in state['submittable_candidate_ids']):
        return tools
    ready = list(state['submittable_candidate_ids'])
    result = []
    for tool in deepcopy(tools):
        if tool['function']['name'] != 'submit_candidate':
            continue
        tool['function']['parameters']['properties']['candidate_id']['enum'] = ready
        result.append(tool)
    return result or tools


def rejection(candidate_id, state):
    return dict(status='rejected', error_code='candidate_image_not_delivered',
        error='Inspect the requested candidate and receive its image before resubmitting.',
        requested_candidate_id=candidate_id, required_inspection_candidate_id=candidate_id,
        submittable_candidate_ids=state['submittable_candidate_ids'],
        suggested_call=dict(name='inspect_trajectory', arguments=dict(candidate_id=candidate_id)))
