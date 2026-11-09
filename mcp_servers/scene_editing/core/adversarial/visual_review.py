"""Visual eligibility and tool availability share one explicit, recoverable ledger."""
from copy import deepcopy


def review_state(nodes, inspections, deliveries, attached, pending_review, manual_limit):
    delivered = sorted({row['candidate_id'] for row in deliveries})
    passed = sorted(key for key, node in nodes.items() if node['result']['status'] == 'reference_screen_passed')
    attached_id = None if attached is None else attached['candidate_id']
    available_images = set(delivered) | ({attached_id} if attached_id else set())
    regular = sum(not row.get('automatic', False) and not row.get('closing_review', False) for row in inspections)
    closing = sum(row.get('closing_review', False) for row in inspections)
    return dict(delivered_candidate_ids=delivered, reference_passed_candidate_ids=passed,
        submittable_candidate_ids=sorted(set(passed) & available_images),
        pending_review_candidate_id=pending_review, attached_candidate_id=attached_id,
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
    if can_inspect:
        item = deepcopy(inspect_tool)
        if first_inspection:
            item['function']['parameters']['properties']['candidate_id']['enum'] = [selected]
            choice = {'type':'function','function':{'name':'inspect_trajectory'}}
        tools.append(item)
    return choice, tools


def rejection(candidate_id, state):
    return dict(status='rejected', error_code='candidate_image_not_delivered',
        error='Inspect the requested candidate and receive its image before resubmitting.',
        requested_candidate_id=candidate_id, required_inspection_candidate_id=candidate_id,
        submittable_candidate_ids=state['submittable_candidate_ids'],
        suggested_call=dict(name='inspect_trajectory', arguments=dict(candidate_id=candidate_id)))
