"""The public route tool uses clear goal parameters and retains old replay calls."""
import sys
from pathlib import Path
import pytest
CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
sys.path.insert(0,str(CORE))
from route_edit_contract import validate,tool
from ego_edit_contract import normalize_edit_call

def request(**changes):
    value=dict(parent_candidate_id='p',start_time_s=4.,left_offset_m=-.3,transition_duration_s=2.,end_behavior='return_to_route',hold_duration_s=.5,return_duration_s=2.,reason='Avoid and return')
    value.update(changes);return value

@pytest.mark.parametrize('changes',[dict(start_time_s=True),dict(left_offset_m=float('nan')),dict(transition_duration_s=0),dict(hold_duration_s=-1),dict(return_duration_s=0),dict(speed_policy='automatic'),dict(preview_only='yes'),dict(end_behavior='hold')])
def test_bad_route_goal_is_rejected(changes):
    with pytest.raises(ValueError):validate(request(**changes))

def test_spatial_direction_and_return_are_expressed_without_coordinate_or_yaw_math():
    normalized=normalize_edit_call('edit_route',request())
    assert normalized['edit_mode']=='route_edit' and normalized['left_offset_m']==-.3
    schema=tool()['function']['parameters']
    assert not {'x_m','y_m','yaw_rad'} & set(schema['properties'])
    assert set(schema['required'])=={'parent_candidate_id','start_time_s','left_offset_m','transition_duration_s','end_behavior','reason'}


def test_keep_offset_has_no_silent_return_and_accepts_explicit_progress_policy():
    value=request(end_behavior='keep_offset',speed_policy='preserve_progress')
    value.pop('hold_duration_s');value.pop('return_duration_s');validate(value)
    with pytest.raises(ValueError):validate(dict(value,return_duration_s=2.))
