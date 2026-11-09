"""Real generator functions retain boundary velocity on irregular native timelines."""
import ast
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


def load_functions(name, namespace):
    tree = ast.parse((CORE/(name+'.py')).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(CORE/(name+'.py')),'exec'),namespace)


@pytest.fixture
def runtime(monkeypatch):
    spec=importlib.util.spec_from_file_location('acceptance_policy',CORE/'acceptance_policy.py')
    policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    monkeypatch.setitem(sys.modules,'acceptance_policy',policy)
    ns=dict(np=np,math=math,policy=policy)
    load_functions('vehicle_trajectory',ns)
    load_functions('vehicle_initial_state',ns)
    load_functions('vehicle_motion',ns)
    load_functions('vehicle_control_search',ns)
    return ns,policy.resolve()


def scene():
    times=np.array([0.,.5,1.,1.5,2.,2.5,3.000015,3.5,4.,4.5,5.])
    xy=np.column_stack([10*times,.1*times])
    return times,xy,np.zeros(len(times))


@pytest.mark.parametrize('generator',['integrate_controls','simulate_motion'])
def test_boundary_velocity_and_prefix_survive_microsecond_adjacent_knot(runtime,generator):
    ns,policy=runtime;times,xy,yaw=scene()
    dense=ns['attack_time_grid'](times)
    if generator=='integrate_controls':
        controls=dict(times_s=[3.,5.],acceleration_mps2=[0.,0.],steering_rad=[0.,0.])
        arrays,_=ns[generator](times,xy,yaw,dense,[4.5,2.,1.5],controls,acceptance=policy)
    else:
        parameters=dict(lateral_start_s=3.,lateral_duration_s=5.,lookahead_m=10.,
            turn_speed_fraction=1.,brake_start_s=100.,stop_time_s=100.,deceleration_mps2=4.5)
        arrays=ns[generator](times,xy,yaw,dense,[4.5,2.,1.5],parameters,xy,yaw,acceptance=policy)
    np.testing.assert_array_equal(arrays[0][times<=3.],xy[times<=3.])
    index=np.flatnonzero(dense==3.)[0]
    velocities=np.diff(arrays[2],axis=0)/np.diff(dense)[:,None]
    assert np.linalg.norm(velocities[index]-velocities[index-1])<1e-3
    acc=np.diff(velocities,axis=0)/((np.diff(dense)[:-1]+np.diff(dense)[1:])/2)[:,None]
    assert np.linalg.norm(acc,axis=1).max()<policy['max_acceleration_mps2']


def test_initial_state_reconstructs_recorded_body_centre_velocity(runtime):
    ns,policy=runtime;times,xy,yaw=scene();yaw=.005*times
    state=ns['initial_state'](times,xy,yaw,[4.5,2.,1.5],3.,policy)
    a=state['yaw'];forward=np.array([np.cos(a),np.sin(a)]);left=np.array([-np.sin(a),np.cos(a)])
    rate=state['speed']/(.6*4.5)*np.tan(state['steering'])
    velocity=state['speed']*forward+(state['lateral_speed_mps']+.3*4.5*rate)*left
    np.testing.assert_allclose(velocity,[10.,.1],atol=1e-10)


def test_initial_slip_remains_subject_to_original_limit(runtime):
    ns,policy=runtime;times,xy,yaw=scene();xy[:,1]=times
    with pytest.raises(ValueError,match='lateral slip exceeds policy'):
        ns['initial_state'](times,xy,yaw,[4.5,2.,1.5],3.,policy)


def test_initial_residual_relaxes_over_native_interval(runtime):
    ns,policy=runtime;times,xy,yaw=scene()
    state=ns['initial_state'](times,xy,yaw,[4.5,2.,1.5],3.,policy)
    assert ns['initial_lateral_velocity'](state,3.)==pytest.approx(.1)
    assert ns['initial_lateral_velocity'](state,3.+state['relaxation_s']/2)==pytest.approx(.05)
    assert ns['initial_lateral_velocity'](state,3.+state['relaxation_s'])==0.


def test_motion_grid_keeps_every_edited_sample_and_native_history(runtime):
    ns,_=runtime;times,xy,_=scene();dense=ns['attack_time_grid'](times)
    values=ns['motion_samples'](times,dense,ns['sample_xy'](times,xy,dense),3.)
    np.testing.assert_array_equal(values[values[:,0]<3.,0],times[times<3.])
    np.testing.assert_array_equal(values[values[:,0]>3.,0],dense[dense>3.])
    assert 3.000015 in values[:,0]


def test_native_history_acceleration_violation_is_still_visible(runtime):
    ns,policy=runtime;times=np.arange(0.,4.1,.5);xy=np.column_stack([5*times**2,np.zeros(len(times))])
    dense=ns['attack_time_grid'](times)
    values=ns['motion_samples'](times,dense,ns['sample_xy'](times,xy,dense),3.)
    velocity=np.diff(values[:,1:],axis=0)/np.diff(values[:,0])[:,None]
    acc=np.diff(velocity,axis=0)/((np.diff(values[:,0])[:-1]+np.diff(values[:,0])[1:])/2)[:,None]
    assert np.linalg.norm(acc,axis=1).max()>policy['max_acceleration_mps2']


def test_edit_boundary_jump_is_not_hidden_by_derivative_grid(runtime):
    ns,policy=runtime;times,xy,_=scene();dense=ns['attack_time_grid'](times)
    points=ns['sample_xy'](times,xy,dense);points[dense>3.,0]+=1.
    values=ns['motion_samples'](times,dense,points,3.)
    velocity=np.diff(values[:,1:],axis=0)/np.diff(values[:,0])[:,None]
    acc=np.diff(velocity,axis=0)/((np.diff(values[:,0])[:-1]+np.diff(values[:,0])[1:])/2)[:,None]
    assert np.linalg.norm(acc,axis=1).max()>1000.
