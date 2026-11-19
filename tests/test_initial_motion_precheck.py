"""Initial trajectory prechecks reject only generator-incompatible states."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.modules.setdefault('openai', SimpleNamespace(OpenAI=object))
from tools.tool04_edit_planner.initial_motion import initial_motion_precheck
from mcp_servers.scene_editing.core.adversarial.initial_motion_precheck import initial_state_precheck


def frames(x_speed=1., y_speed=0., yaw_rate=0.):
    return [
        dict(timestamp=0, anns=dict(track_tokens=['car'], gt_boxes=[[0., 0., 0., 4., 2., 1.5, 0.]])),
        dict(timestamp=1, anns=dict(track_tokens=['car'], gt_boxes=[[x_speed, y_speed, 0., 4., 2., 1.5, yaw_rate]])),
    ]


def test_stdlib_precheck_accepts_forward_motion():
    result = initial_motion_precheck(frames(), 'car')
    assert result['passed'] and result['forward_speed_mps'] == 1.


def test_stdlib_precheck_rejects_reverse_and_excess_steering():
    assert 'reversing' in initial_motion_precheck(frames(x_speed=-1), 'car')['errors'][0]
    assert 'steering' in initial_motion_precheck(frames(yaw_rate=2), 'car')['errors'][-1]


def test_microsecond_timestamps_produce_the_same_lateral_slip_rejection():
    seconds=frames(x_speed=5.,y_speed=.5)
    microseconds=[dict(frame,timestamp=frame['timestamp']*1_000_000) for frame in seconds]
    first=initial_motion_precheck(seconds,'car')
    second=initial_motion_precheck(microseconds,'car')
    assert not first['passed'] and not second['passed']
    assert second['errors']==first['errors']
    assert second['rear_axle_lateral_speed_mps']==first['rear_axle_lateral_speed_mps']==.5
    assert second['first_violation_time_s']==first['first_violation_time_s']==1.


def test_lidar_boxes_are_checked_in_initial_ego_coordinates():
    import math
    def transform(x,y,yaw):
        return [[math.cos(yaw),-math.sin(yaw),0.,x],[math.sin(yaw),math.cos(yaw),0.,y],
                [0.,0.,1.,0.],[0.,0.,0.,1.]]
    origin=transform(100.,50.,.4)
    data=[]
    for index in range(2):
        data.append(dict(timestamp=index,lidar2global=transform(110.+index*math.cos(.4),50.+index*math.sin(.4),.4),ego2global=origin,
            anns=dict(track_tokens=['car'],gt_boxes=[[0.,0.,0.,4.,2.,1.5,0.]])))
    result=initial_motion_precheck(data,'car')
    assert result['passed'] and abs(result['forward_speed_mps']-1.)<1e-12


def editor(x_speed):
    return SimpleNamespace(
        target=dict(operation='edit_existing', track_id='car', size_m=[4., 2., 1.5]),
        times=np.array([0., 1.]), seed_xy=np.array([[0., 0.], [x_speed, 0.]]),
        seed_yaw=np.array([0., 0.]), edit_start_s=0., constraints={},
    )


def test_backend_precheck_matches_initial_state_policy():
    assert initial_state_precheck(editor(1)) is None
    result = initial_state_precheck(editor(-1))
    assert result['stop_reason'] == 'unsupported_initial_state'
    assert 'reversing' in result['reason']


@pytest.mark.parametrize('speed', [0., 1., -1.])
def test_fixed_seed_is_independent_of_moving_rotating_ego(speed):
    value = editor(speed)
    expected = initial_state_precheck(value)
    value.frames = []
    for index, angle in enumerate([.4, .9]):
        transform = np.eye(4)
        transform[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                              [np.sin(angle), np.cos(angle)]]
        transform[:2, 3] = [100.+10*index, 50.+3*index]
        value.frames.append(dict(ego2global=transform))
    assert initial_state_precheck(value) == expected


def test_valid_initial_state_does_not_reject_editable_recorded_future():
    value = editor(1.)
    value.times = np.array([0., 1., 2., 3.])
    value.seed_xy = np.array([[0., 0.], [1., 0.], [0., 2.], [-1., 4.]])
    value.seed_yaw = np.zeros(4)
    assert initial_state_precheck(value) is None
    value.edit_start_s = 1.5
    result = initial_state_precheck(value)
    assert 'reversing' in result['reason']
    assert result['measurements']['boundary_s'] == 1.5
    assert result['measurements']['coordinate_frame'] == 'initial_ego'


@pytest.mark.parametrize('offset', [0., 50.])
@pytest.mark.parametrize('motion, reason', [
    ((-.1, 0.), 'reversing'),
    ((1., .4), 'lateral slip'),
    ((16., 0.), 'speed exceeds'),
])
def test_backend_keeps_generator_limits_without_a_distance_exemption(offset, motion, reason):
    value = editor(motion[0])
    value.seed_xy[1, 1] = motion[1]
    value.seed_xy[:, 0] += offset
    result = initial_state_precheck(value)
    assert result['stop_reason'] == 'unsupported_initial_state'
    assert reason in result['reason']


def test_backend_uses_seed_heading_and_actual_boundary_steering():
    value = editor(1.)
    value.seed_xy = np.array([[4., 1.], [4., 2.]])
    value.seed_yaw[:] = np.pi/2
    assert initial_state_precheck(value) is None
    # The first interval implies yaw rate but no rear-axle slip at t=0;
    # steering itself must still remain within the generator's limits.
    value.seed_xy = np.array([[0., 0.], [.1, 1.2]])
    value.seed_yaw = np.array([0., 1.])
    assert 'steering' in initial_state_precheck(value)['reason']


def test_backend_does_not_disguise_unrelated_generator_errors(monkeypatch):
    import vehicle_initial_state
    def broken(*args):
        raise ValueError('unexpected malformed input')
    monkeypatch.setattr(vehicle_initial_state, 'initial_state', broken)
    with pytest.raises(ValueError, match='unexpected malformed input'):
        initial_state_precheck(editor(1.))


def test_added_vehicle_does_not_use_recorded_actor_precheck():
    value = editor(-1.)
    value.target['operation'] = 'add_vehicle'
    assert initial_state_precheck(value) is None
