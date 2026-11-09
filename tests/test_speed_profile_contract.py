"""One public speed tool; explicit goals and legacy replay have separate contracts."""
import copy
import ast
import sys
from pathlib import Path
import pytest
CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
sys.path.insert(0,str(CORE))
from speed_profile_contract import validate,tool
from ego_edit_contract import mode_tools,normalize_edit_call
from mcp_servers.scene_editing.core.adversarial.defense_generation_contract import validate_request,normalize_request


def test_effective_agent_prompt_describes_only_current_editing_entries():
    # Evaluate the actual prompt assembly without importing the scientific runtime.
    source=ast.parse((CORE/'run_ego_editor_agent.py').read_text())
    assignments=[node for node in source.body if
        (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT' for t in node.targets)) or
        (isinstance(node,ast.AugAssign) and isinstance(node.target,ast.Name) and node.target.id=='PROMPT')]
    namespace={}
    exec(compile(ast.Module(body=assignments,type_ignores=[]),'<agent-prompt>','exec'),namespace)
    prompt=namespace['PROMPT']
    names={item['function']['name'] for item in mode_tools(dict(function=dict(parameters=dict(properties={}))))}
    assert names=={'edit_speed_profile','edit_route'}
    assert all(name in prompt for name in names)
    assert 'decision_summary' in prompt and 'recommended_previews' in prompt
    hidden={'edit_trajectory_points','set_trajectory_offsets','replace_trajectory_positions',
            'edit_path_progress','brake_along_path','adjust_trajectory_speed'}
    assert not any(name in prompt for name in hidden)
    visual=ast.parse((CORE/'run_ego_visual_editor_agent.py').read_text())
    request_strings=[node.value for node in ast.walk(visual) if isinstance(node,ast.Constant) and isinstance(node.value,str)]
    assert not any(name in text for name in hidden for text in request_strings)


def request(**changes):
    d=dict(parent_candidate_id='p',operation='set_speed_targets',start_time_s=3.,
        targets=[dict(time_s=6.,speed_mps=0.)],reason='Preserve route, repair stop')
    d.update(changes);return d


@pytest.mark.parametrize('changes',[dict(start_time_s=True),dict(targets=[]),dict(targets=[dict(time_s=6.,speed_mps=float('nan'))]),
    dict(targets=[dict(time_s=6.,speed_mps=0.,x_m=1.)]),dict(braking_start_shift_s=.2),dict(max_acceleration_mps2=float('inf')),
    dict(preview_only='true'),dict(path_source='invented_path'),dict(operation='teleport')])
def test_bad_or_conflicting_speed_requests_are_rejected(changes):
    with pytest.raises(ValueError):validate(request(**changes))


def test_explicit_retime_and_reverse_states_do_not_need_redundant_intent_fields():
    validate(dict(parent_candidate_id='p',operation='retime_braking',braking_start_shift_s=.2,preserve='stop_position',reason='Delay onset'))
    args=request(targets=[dict(time_s=6.,speed_mps=0.),dict(time_s=7.,speed_mps=-1.)])
    normalized=normalize_edit_call('edit_speed_profile',args)
    assert normalized['edit_mode']=='speed_schedule' and 'edit_mode' not in args
    assert set(tool()['function']['parameters']['required'])=={'parent_candidate_id','operation','reason'}


@pytest.mark.parametrize('name',['edit_trajectory_points','brake_along_path','adjust_trajectory_speed','set_trajectory_offsets','replace_trajectory_positions','edit_path_progress'])
def test_removed_editing_entries_are_rejected(name):
    with pytest.raises(ValueError,match='Unsupported editing tool'):
        normalize_edit_call(name, {})

def test_public_tools_only_expose_current_editing_entries():
    assert {item['function']['name'] for item in mode_tools({})} == {'edit_route','edit_speed_profile'}


def test_foci_uses_signed_timed_target_semantics_and_preserves_input():
    args=dict(parent_candidate_id='p',edit_window_s=[3.,11.5],max_candidates=1,
        speed_targets=[dict(time_s=8.,speed_mps=0.),dict(time_s=11.5,speed_mps=-1.)])
    before=copy.deepcopy(args)
    normalized=normalize_request('optimize_defense_candidate',args,11.499597)
    assert normalized['speed_targets'][-1]['time_s']==11.499597 and args==before
    validate_request('optimize_defense_candidate',normalized,boundary=3.,end=11.499597,remaining=1)
    with pytest.raises(ValueError):validate_request('optimize_defense_candidate',dict(args,speed_targets=[dict(time_s=2.,speed_mps=0.)]),boundary=3.,end=11.5,remaining=1)
