"""A checked refinement precedes final choice; candidate source never fixes the parent."""
VERSION='refine_then_compare.v1'
GUIDANCE=(
    'Review the initial candidates before deciding whether an edit is useful. A reviewed passing '
    'initial candidate may be submitted directly when it satisfies the task; no edit is required '
    'merely to prove search effort. When a concrete improvement is needed, choose a promising '
    'candidate from ANY source and execute a supported trajectory edit, repair or parent-based '
    'local FOCI. Preserve passing alternatives while trying the change. After a changed trajectory '
    'has been checked and its diagram reviewed, compare its actual progress, reserve, motion, '
    'road/other-actor checks and terminal behavior against its parent and the retained alternatives. '
    'Submit the most suitable reviewed passing result, including an UNCHANGED fallback when an '
    'attempted edit fails or is worse. Do not produce a no-op or keep editing solely to spend '
    'budget. Resource limits remain binding; do not claim an edit was tested if unavailable.')


def refinement_state(editor,remaining,turns_remaining,enabled,reviewed_candidate_ids=None):
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
    reviewed_initial=[] if reviewed_candidate_ids is None else [cid for cid,node in editor.nodes.items()
        if node.get('result',{}).get('status')=='reference_screen_passed'
        and cid in reviewed_candidate_ids]
    eligible=enabled and not changed and not reviewed_initial and remaining>0 and turns_remaining>=2
    unreviewed=[] if reviewed_candidate_ids is None else [cid for cid in changed
        if editor.nodes[cid].get('result',{}).get('status')=='reference_screen_passed'
        and cid not in reviewed_candidate_ids]
    status=('disabled' if not enabled else 'comparison_review_pending' if unreviewed else 'checked_refinement_available' if changed else
            'reviewed_initial_candidate_available' if reviewed_initial else 'refinement_pending' if eligible else 'not_attempted_resource_limit')
    return dict(version=VERSION,status=status,pending=eligible or (enabled and bool(unreviewed)),
        checked_changed_candidate_ids=changed,unchanged_attempt_candidate_ids=unchanged,
        unreviewed_passing_refinement_ids=unreviewed,
        retained_initial_candidate_id=getattr(editor,'selected_initial',getattr(editor,'current',None)),
        new_candidate_budget_remaining=remaining,model_actions_remaining=turns_remaining,
        final_choice='Compare the latest checked refinement with the retained fallback using candidate_differences and the attached/delivered diagrams. State the concrete benefit and cost of your choice and any missing evidence. Any reviewed passing candidate remains eligible; greater clearance alone is not a reason to discard useful progress when reserve is comparable. No preferred source.')
