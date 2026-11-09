"""Deterministic ego trajectory generation from a structured defense decision."""
from dataclasses import dataclass

import numpy as np

from ego_avoidance_contract import validate_decision


@dataclass(frozen=True)
class EgoTrajectory:
    times_s:np.ndarray
    frame_ids:np.ndarray
    xy:np.ndarray
    z_m:np.ndarray
    heading_rad:np.ndarray
    speed_mps:np.ndarray
    decision:dict


def smoothstep(value):
    value=np.clip(value,0.,1.)
    return value**3*(10.+value*(-15.+6.*value))


def smoothstep_derivative(value,duration):
    value=np.asarray(value,dtype=float)
    inside=(value>0)&(value<1)
    result=np.zeros_like(value)
    result[inside]=30.*value[inside]**2*(1.-value[inside])**2/duration
    return result


def _unique_path(times,xy,heading):
    segment=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    distance=np.concatenate([[0.],np.cumsum(segment)])
    keep=np.concatenate([[True],np.diff(distance)>1e-9])
    if keep.sum()<2:
        raise ValueError('Ego baseline has no usable forward path')
    return distance,keep,np.unwrap(heading)


def _integrate_braking(times,baseline_times,baseline_distance,baseline_speed,start,deceleration,target):
    initial_speed=float(np.interp(start,baseline_times,baseline_speed))
    if target>initial_speed+1e-9:
        raise ValueError('Brake target speed exceeds speed at braking start')
    # Quintic speed transition: peak deceleration equals the decision value and
    # both acceleration and jerk are zero at the transition boundaries.
    duration=1.875*(initial_speed-target)/deceleration if initial_speed>target else 0.
    work=np.unique(np.concatenate([baseline_times,[start,start+duration]]))
    speed=np.interp(work,baseline_times,baseline_speed)
    after=work>start
    speed[after]=initial_speed-(initial_speed-target)*smoothstep((work[after]-start)/duration) if duration else target
    distance=np.interp(work,baseline_times,baseline_distance)
    start_index=int(np.flatnonzero(np.isclose(work,start,rtol=0,atol=1e-12))[0])
    for i in range(start_index+1,len(work)):
        distance[i]=distance[i-1]+.5*(speed[i-1]+speed[i])*(work[i]-work[i-1])
    return np.interp(times,work,distance),np.interp(times,work,speed)


def generate_ego_trajectory(scene,decision):
    """Follow the recorded path with new longitudinal progress and a C2 lateral offset."""
    decision=validate_decision(decision)
    scene=scene.planning_view() if len(scene.times_s)>1 and np.isclose(scene.times_s[1]-scene.times_s[0],.05) else scene
    times=np.asarray(scene.times_s,dtype=float)
    if decision['planning_horizon_s']+1e-9<times[-1] or decision['start_s']>=times[-1]:
        raise ValueError('Decision does not cover the frozen planning window')
    xy=np.asarray(scene.ego_xy,dtype=float);heading=np.asarray(scene.ego_heading_rad,dtype=float)
    distance,keep,unwrapped=_unique_path(times,xy,heading)
    path_distance=distance[keep];path_xy=xy[keep];path_heading=unwrapped[keep]
    action=decision['action_type'];brake=decision.get('brake');lane=decision.get('lane_change')
    early=times<=min(times[0]+1.,times[-1])
    nominal_speed=float(np.median(np.asarray(scene.ego_speed_mps,dtype=float)[early]))
    if not np.isfinite(nominal_speed) or nominal_speed<=0:
        raise ValueError('Ego baseline has no positive initial speed estimate')
    nominal_speeds=np.full(len(times),nominal_speed)
    nominal_distance=distance[0]+nominal_speed*(times-times[0])
    if brake:
        new_distance,longitudinal_speed=_integrate_braking(times,times,nominal_distance,nominal_speeds,
            decision['start_s'],brake['deceleration_mps2'],brake['target_speed_mps'])
    else:
        new_distance=nominal_distance;longitudinal_speed=nominal_speeds
    base=np.column_stack([np.interp(new_distance,path_distance,path_xy[:,axis]) for axis in range(2)])
    base_heading=np.interp(new_distance,path_distance,path_heading)
    offset=np.zeros(len(times));offset_rate=np.zeros(len(times))
    if lane:
        lane_start=lane['start_s'];duration=lane['duration_s']
        if lane_start+duration>times[-1]+1e-9:
            raise ValueError('Lane change exceeds the frozen planning window')
        phase=(times-lane_start)/duration
        offset=lane['offset_m']*smoothstep(phase)
        offset_rate=lane['offset_m']*smoothstep_derivative(phase,duration)
    tangent=np.column_stack([np.cos(base_heading),np.sin(base_heading)])
    normal=np.column_stack([-tangent[:,1],tangent[:,0]])
    result_xy=base+normal*offset[:,None]
    velocity=tangent*longitudinal_speed[:,None]+normal*offset_rate[:,None]
    speed=np.linalg.norm(velocity,axis=1)
    result_heading=base_heading.copy();moving=speed>=.5
    result_heading[moving]=np.arctan2(velocity[moving,1],velocity[moving,0])
    result_heading=np.unwrap(result_heading)
    if not np.isfinite(result_xy).all() or np.any(np.diff(new_distance)<-1e-8):
        raise ValueError('Generated ego trajectory is non-finite or reverses along the baseline')
    return EgoTrajectory(times.copy(),scene.frame_ids.copy(),result_xy,scene.ego_z_m.copy(),result_heading,speed,decision)
