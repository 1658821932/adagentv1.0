"""Braking proposals depend on scene state and replay through the real speed tool."""
from pathlib import Path
import sys
import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
if str(CORE) not in sys.path:sys.path.insert(0,str(CORE))
import acceptance_policy as policy
from adaptive_braking import proposals,verify_constructor
from ego_trajectory_edit import modern_edit


def case():
    times=np.arange(0.,8.01,.5);xy=np.column_stack([8.5*times,np.zeros(len(times))]);yaw=np.zeros(len(times))
    effective=policy.for_new_run();effective.update(edit_start_s=1.5,allow_safe_stop=True)
    return times,xy,yaw,effective


def test_scene_derived_stop_requests_replay_and_preserve_history():
    times,xy,yaw,effective=case()
    options=proposals(times,xy,yaw,{'edit_start_clearance_m':6.85},effective,True)
    assert 1<len(options)<=5
    for mode,start,args in options:
        request=dict(args,edit_mode=mode,parent_candidate_id='recorded_baseline',start_time_s=start,
            reason='fixture',speed_schedule_version='state_continuity.v4')
        xx,aa,curve=modern_edit(times,lambda key:(xy,yaw),request,1.5,acceptance=effective)
        assert curve['max_acceleration_mps2']<=6.+1e-8
        np.testing.assert_array_equal(xx[times<=1.5],xy[times<=1.5])
        assert verify_constructor(times,xy,yaw,xx,aa,request,1.5,effective)==curve
        broken=xx.copy();broken[-1,0]+=.1
        with pytest.raises(ValueError,match='differs'):
            verify_constructor(times,xy,yaw,broken,aa,request,1.5,effective)


def test_no_stop_permission_or_no_gap_does_not_invent_a_passing_stop():
    times,xy,yaw,effective=case()
    assert proposals(times,xy,yaw,{'edit_start_clearance_m':6.85},effective,False)==[]
    assert proposals(times,xy,yaw,{'edit_start_clearance_m':0.},effective,True)==[]
    close=proposals(times,xy,yaw,{'edit_start_clearance_m':6.85},effective,True)
    far=proposals(times,xy,yaw,{'edit_start_clearance_m':25.},effective,True)
    assert [row[2]['targets'] for row in close]!=[row[2]['targets'] for row in far]
