"""A checked refinement precedes final choice; candidate source never fixes the parent."""
VERSION='refine_then_compare.v1'
GUIDANCE=(
    'Your task includes editing, not only choosing a ready-made trajectory. Treat initial selection '
    'as a working seed/fallback, not the final answer. Choose a promising existing candidate from '
    'ANY source, identify a concrete improvement and execute a supported trajectory edit, repair '
    'or parent-based local FOCI. The edit reason must state the benefit to preserve, the defect to '
    'change and the tradeoff/checks to revisit. A further constructor pool alone does not satisfy '
    'this task. Preserve passing alternatives while trying the change. '
    'After a changed trajectory has been checked and its diagram reviewed, compare its actual '
    'progress, reserve, motion, road/other-actor checks and terminal behavior against its parent and '
    'the retained alternatives. Submit the most suitable reviewed passing result, including an '
    'UNCHANGED fallback when the attempted edit fails or is worse. An attempted improvement need '
    'not succeed, and the final candidate need not be an edit. Do not produce a no-op to unlock '
    'submission or keep editing solely to spend budget. You may inspect another candidate or edit '
    'a different parent from the initial selection. Resource limits remain binding; do not claim '
    'an edit was tested if the available budget prevented it.')


def refinement_state(editor,remaining,turns_remaining,enabled):
    changed=[];unchanged=[]
    tolerance=(getattr(editor,'effective_policy',{}) or {}).get('pose_tolerance',1e-6)
    for cid,node in editor.nodes.items():
        operation=node.get('operation') or {}
        if operation.get('mode') not in ('point_edit','repair','foci') or operation.get('generation_source')=='constructor':
            continue
        parent=editor.nodes.get(node.get('parent_id'))
        if parent is None or 'xy' not in node or 'xy' not in parent:
            continue
        moved=any(abs(float(a)-float(b))>tolerance for p,q in zip(node['xy'],parent['xy']) for a,b in zip(p,q))
        turned=any(abs(float(a)-float(b))>tolerance for a,b in zip(node.get('yaw',[]),parent.get('yaw',[])))
        (changed if moved or turned else unchanged).append(cid)
    eligible=enabled and not changed and remaining>0 and turns_remaining>=2
    status=('disabled' if not enabled else 'checked_refinement_available' if changed else
            'refinement_pending' if eligible else 'not_attempted_resource_limit')
    return dict(version=VERSION,status=status,pending=eligible,
        checked_changed_candidate_ids=changed,unchanged_attempt_candidate_ids=unchanged,
        retained_initial_candidate_id=getattr(editor,'selected_initial',getattr(editor,'current',None)),
        new_candidate_budget_remaining=remaining,model_actions_remaining=turns_remaining,
        final_choice='Any reviewed passing candidate, including the original fallback, after real edit feedback; no preferred source.')
