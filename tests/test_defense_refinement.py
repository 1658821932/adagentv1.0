"""Real edits unlock comparison; constructor-only/no-op actions do not, fallback stays valid."""
from types import SimpleNamespace
from mcp_servers.scene_editing.core.adversarial.defense_refinement import refinement_state


def editor():
    return SimpleNamespace(effective_policy={'pose_tolerance':1e-6},selected_initial='initial',
        nodes={'initial':dict(xy=[[0.,0.],[1.,0.]],yaw=[0.,0.],parent_id=None)})


def child(owner,name,source=None,changed=True,passed=True):
    operation=dict(mode='point_edit')
    if source:operation['generation_source']=source
    owner.nodes[name]=dict(parent_id='initial',operation=operation,xy=[[0.,0.],[2. if changed else 1.,0.]],
        yaw=[0.,0.],result=dict(status='reference_screen_passed' if passed else 'reference_screen_failed'))


def test_failed_checked_edit_unlocks_original_fallback_but_noop_and_pool_do_not():
    e=editor()
    assert refinement_state(e,5,8,True)['pending']
    child(e,'constructor_output','constructor')
    child(e,'unchanged',changed=False)
    state=refinement_state(e,3,6,True)
    assert state['pending'] and state['unchanged_attempt_candidate_ids']==['unchanged']
    child(e,'failed_edit',passed=False)
    state=refinement_state(e,2,5,True)
    assert not state['pending'] and state['checked_changed_candidate_ids']==['failed_edit']
    assert state['retained_initial_candidate_id']=='initial'


def test_parent_based_foci_counts_and_resource_waivers_are_explicit():
    e=editor();child(e,'foci_edit',source='foci')
    e.nodes['foci_edit']['operation']['mode']='foci'
    assert refinement_state(e,4,7,True)['status']=='checked_refinement_available'
    assert refinement_state(editor(),0,7,True)['status']=='not_attempted_resource_limit'
    assert refinement_state(editor(),5,1,True)['status']=='not_attempted_resource_limit'
    assert refinement_state(editor(),5,7,False)['status']=='disabled'


def test_reviewed_passing_initial_candidate_does_not_require_artificial_edit():
    e=editor()
    e.nodes['initial']['result']=dict(status='reference_screen_passed')
    state=refinement_state(e,5,8,True,reviewed_candidate_ids=['initial'])
    assert not state['pending'] and state['status']=='reviewed_initial_candidate_available'
    assert refinement_state(e,5,8,True)['pending']
    e.nodes['initial']['result']['status']='reference_screen_failed'
    assert refinement_state(e,5,8,True,reviewed_candidate_ids=['initial'])['pending']


def test_checked_is_not_reviewed_but_attached_image_allows_same_turn_comparison():
    e=editor();child(e,'foci_edit',source='foci')
    state=refinement_state(e,4,7,True,reviewed_candidate_ids=['initial'])
    assert state['status']=='comparison_review_pending' and state['pending']
    assert state['unreviewed_passing_refinement_ids']==['foci_edit']
    ready=refinement_state(e,4,7,True,reviewed_candidate_ids=['initial','foci_edit'])
    assert not ready['pending'] and ready['retained_initial_candidate_id']=='initial'
    failed=editor();child(failed,'failed',passed=False)
    assert not refinement_state(failed,4,7,True,reviewed_candidate_ids=['initial'])['pending']
