"""No half-second takeover drift or frozen moving boxes between native samples."""
from copy import deepcopy
import math
import pytest
from mcp_servers.scene_editing.core.adversarial.pdm_temporal import align_takeover,RecordedActorReplay


def test_forward_microsecond_native_boundary_is_used_without_stale_state():
    stamps=[0,500000,1000000,1500000,2000000,2500151,3000011,3500000]
    result=align_takeover(stamps,3.)
    assert result['index']==6 and result['takeover_time_us']==3000011
    assert result['alignment_delay_s']==pytest.approx(.000011)
    assert result['extrapolation_s']==0.


def test_alignment_never_takes_control_before_requested_boundary():
    result=align_takeover([0,1000000,2000000,2999544,3500000],3.)
    assert result['takeover_time_us']==3000000 and result['index']==3
    assert result['extrapolation_s']==pytest.approx(.000456)


def test_distant_future_observation_is_not_added_to_history():
    result=align_takeover([0,1000000,2500000,3200000,4000000],3.)
    assert result['index']==2 and result['takeover_time_us']==3000000


def actor(track='car',x=0.,yaw=0.):
    return dict(track=track,label='vehicle',x=x,y=0.,yaw=yaw,length=4.,width=2.,height=1.5)


def test_interpolated_truth_moves_the_recorded_actor_without_mutating_input():
    observations=[[actor(x=0.)],[actor(x=5.)]];before=deepcopy(observations)
    replay=RecordedActorReplay([0.,.5],observations)
    sample,coverage=replay.sample(.4)
    assert sample[0]['x']==pytest.approx(4.) and coverage['interpolated_tracks']==1
    assert observations==before
    assert replay.sample(0.)[0]==observations[0]


def test_missing_pairs_remain_explicit_and_future_only_tracks_do_not_appear():
    replay=RecordedActorReplay([0.,.5],[[actor('old')],[actor('new',x=5.)]])
    sample,coverage=replay.sample(.4)
    assert [obj['track'] for obj in sample]==['old'] and coverage['held_missing_pairs']==1


def test_heading_interpolation_uses_shortest_rotation():
    replay=RecordedActorReplay([0.,.5],[[actor(yaw=math.pi-.1)],[actor(yaw=-math.pi+.1)]])
    assert replay.sample(.25)[0][0]['yaw']==pytest.approx(math.pi)


def test_replay_does_not_claim_coverage_outside_the_clip():
    replay=RecordedActorReplay([0.,.5],[[actor()],[actor()]])
    with pytest.raises(ValueError,match='coverage'):replay.sample(.6)
