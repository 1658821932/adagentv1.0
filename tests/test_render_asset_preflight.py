"""Asset quality is checked without GPU work and without allowing extrapolation."""
import numpy as np
import pytest
from mcp_servers.scene_editing.core.adversarial.render_asset_preflight import assess_replay


def test_late_annotation_fails_coverage_instead_of_extrapolating_asset():
    result=assess_replay([0,1],[0,1500000],[[0.,0.],[1.5,0.]],[0,1000000],[[0.,0.,0.],[1.,0.,0.]])
    assert result['status']=='failed'
    assert result['frames'][1]['reason']=='outside_asset_replay_coverage'
    assert result['frames'][1]['residual_m'] is None


def test_same_half_metre_alignment_limit_is_preserved():
    result=assess_replay([17,18],[0,1000000],[[.49,0.],[1.58,0.]],[0,1000000],[[0.,0.,0.],[1.,0.,0.]])
    assert result['frames'][0]['passed'] and not result['frames'][1]['passed']
    assert result['frames'][1]['reason']=='pose_annotation_mismatch'


def test_large_absolute_timestamps_interpolate_using_small_relative_times():
    base=1631821502000280
    result=assess_replay([0],[base+500000],[[.5,0.]],[base,base+1000000],[[0.,0.,0.],[1.,0.,0.]])
    assert result['status']=='passed' and result['frames'][0]['residual_m']==0.


def test_static_asset_still_checks_every_requested_annotation():
    result=assess_replay([0,1],[0,2000000],[[1.,2.],[2.,2.]],None,np.array([1.,2.,.5]))
    assert result['frames'][0]['passed'] and not result['frames'][1]['passed']


def test_missing_annotation_cannot_silently_drop_requested_frame():
    with pytest.raises(ValueError,match='Every requested frame'):
        assess_replay([0,1],[0,1000000],[[0.,0.]],None,np.zeros(3))


def test_absolute_placement_does_not_add_source_pose_drift():
    from mcp_servers.scene_editing.core.adversarial.render_actor_pose import grounded_absolute_pose
    class Road:
        def height_with_local_plane(self,x,y):return dict(height=.8)
    transform=np.eye(4);transform[:2,3]=[100,200]
    position,heading,_=grounded_absolute_pose([10,2],.4,transform,Road(),2.)
    np.testing.assert_allclose(position,[110,202,1.8])
    assert heading==.4
