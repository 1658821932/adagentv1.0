"""Shared ego history owned by attack editing; no policy actions after takeover."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from run_ego_reference import ego_path_from_frames, observed_objects_from_frames, EGO_FRONT, EGO_REAR, EGO_WIDTH, polygon

TAKEOVER_S = 1.5


def target_schema():
    return dict(type='array',maxItems=4,description='Optional ego speed targets during 0–1.5s. Omit to keep the current shared history; [] restores the source history. Along the original route only; preserve t=0 pose and speed.',
        items=dict(type='object',properties=dict(time_s=dict(type='number',exclusiveMinimum=0,maximum=1.5),
            speed_mps=dict(type='number',minimum=0)),required=['time_s','speed_mps'],additionalProperties=False))


def source_scene(scene):
    scene=Path(scene).resolve()
    manifest=scene/'ego-history.json'
    if manifest.is_file():
        data=json.loads(manifest.read_text())
        source=Path(data['source_scene']).resolve()
        actual=hashlib.sha256((source/'metadata/frames.json').read_bytes()).hexdigest()
        if actual!=data['source_frames_sha256']:raise ValueError('Ego history source changed')
        return source
    from official_map_coverage import SOURCE_BINDING
    binding=scene/SOURCE_BINDING
    if binding.is_file():
        source=Path(json.loads(binding.read_text())['source_scene']).resolve()
        if not (scene/'metadata/frames.json').samefile(source/'metadata/frames.json'):
            raise ValueError('Expanded map scene does not share the original frame data')
        return source_scene(source)
    return scene


def generate(frames, targets, acceptance):
    """Reuse the spatial speed editor; return a continuous, bounded source-route history."""
    from ego_trajectory_edit import start_state
    from ego_speed_schedule import Route
    times,xy,yaw=ego_path_from_frames(frames)
    if times[-1]<TAKEOVER_S:raise ValueError('Source does not cover the 1.5s takeover')
    if not 1<=len(targets)<=4:raise ValueError('Provide 1–4 ego speed targets, ending at 1.5s')
    knots=[0.];speeds=[start_state(times,xy,yaw,0.)['signed_speed_mps']]
    if speeds[0]<0:raise ValueError('This history tool supports forward motion only')
    for point in targets:
        if set(point)!={'time_s','speed_mps'}:raise ValueError('Ego targets use time_s and speed_mps')
        t,v=point['time_s'],point['speed_mps']
        if type(t) not in (int,float) or type(v) not in (int,float) or not np.isfinite([t,v]).all():raise ValueError('Ego targets must be finite')
        if not knots[-1]<t<=TAKEOVER_S or not 0<=v<=acceptance['max_speed_mps']:raise ValueError('Ego speed target is outside the history or speed limit')
        required=1.5*abs(v-speeds[-1])/(t-knots[-1])
        if required>acceptance['max_acceleration_mps2']+1e-8:
            raise ValueError(f'Ego interval {knots[-1]:.2f}–{t:.2f}s needs peak acceleration {required:.2f}m/s², limit {acceptance["max_acceleration_mps2"]:.2f}; reduce speed change or allow more time')
        knots.append(float(t));speeds.append(float(v))
    if abs(knots[-1]-TAKEOVER_S)>1e-9:raise ValueError('Last ego speed target must be at 1.5s')
    state=start_state(times,xy,yaw,0.)
    # Only the contiguous observed forward route is needed for this history.
    # A later reverse segment (including a stopped-tail positional drift) is
    # not another forward path to fit through. Do not extrapolate past it.
    midpoint_yaw=(yaw[:-1]+yaw[1:])/2
    signed=np.sum(np.diff(xy,axis=0)*np.column_stack([np.cos(midpoint_yaw),np.sin(midpoint_yaw)]),axis=1)
    reverse=np.flatnonzero(signed < -1e-6)
    route_count=int(reverse[0])+1 if len(reverse) else len(times)
    if route_count<2:raise ValueError('route_unavailable: no observed forward route for ego history')
    route_times=times[:route_count];route_xy=xy[:route_count];route_yaw=yaw[:route_count]
    route=Route(route_times,lambda key:(route_xy,route_yaw),'recorded_baseline',state,None,'parent')
    prefix_law=CubicHermiteSpline(knots,speeds,np.zeros(len(knots)))
    distance=float(prefix_law.integrate(0.,TAKEOVER_S))
    remaining=route.distance[-1]-route.origin-distance
    if remaining<1e-6:raise ValueError('Ego history exhausts the known route; no extrapolation')
    # This is only a continuous reference continuation. VAD/experts own the
    # executed future. Keep the navigation goal separately from this timing.
    if speeds[-1]>1e-8:
        stop_time=TAKEOVER_S+2*remaining/speeds[-1]
        if 1.5*speeds[-1]/(stop_time-TAKEOVER_S)>acceptance['max_acceleration_mps2']+1e-8:
            raise ValueError('Too little known route for a continuous reference continuation')
        knots.append(float(stop_time));speeds.append(0.)
    if knots[-1]<times[-1]:knots.append(float(times[-1]));speeds.append(0.)
    law=CubicHermiteSpline(knots,speeds,np.zeros(len(knots)))
    query=np.unique(np.r_[times,np.linspace(0.,times[-1],int(np.ceil(times[-1]/.025))+1),knots])
    query=query[query<=times[-1]+1e-9]
    positions,velocities,angles=route.drive(query,law)
    angles=np.unwrap(angles)
    angles+=np.arctan2(np.sin(yaw[0]-angles[0]),np.cos(yaw[0]-angles[0]))
    positions[0]=xy[0]
    mask=query<=TAKEOVER_S+1e-8
    acceleration=np.gradient(velocities,query,axis=0)
    if np.max(np.linalg.norm(acceleration[mask],axis=1))>acceptance['max_acceleration_mps2']+.02:
        raise ValueError('Curved ego history exceeds planar acceleration limit')
    speed=np.linalg.norm(velocities,axis=1)
    tangent=velocities/np.maximum(speed[:,None],1e-8)
    along=np.sum(acceleration*tangent,axis=1)
    lateral=np.sqrt(np.maximum(0.,np.sum(acceleration**2,axis=1)-along**2))
    if np.max(lateral[mask])>acceptance['max_lateral_acceleration_mps2']+.02:
        raise ValueError('Curved ego history exceeds lateral acceleration limit')
    if np.max(abs(np.gradient(np.unwrap(angles),query)[mask]))>acceptance['max_yaw_rate_radps']+.02:
        raise ValueError('Ego history exceeds yaw-rate limit')
    check_history_actors(frames, times, query[mask], positions[mask], angles[mask])
    native_xy=np.column_stack([np.interp(times,query,positions[:,i]) for i in range(2)])
    native_yaw=np.interp(times,query,np.unwrap(angles))
    native_xy[0]=xy[0];native_yaw[0]=yaw[0]
    index=int(np.searchsorted(times,TAKEOVER_S+.001,side='right'))-1
    centers=native_xy+(EGO_FRONT-EGO_REAR)/2*np.column_stack([np.cos(native_yaw),np.sin(native_yaw)])
    native_step=round(float(np.median(np.diff(times)))/.05)*.05
    return native_xy,native_yaw,dict(targets=deepcopy(targets),takeover_s=TAKEOVER_S,
        takeover_xy_m=[float(np.interp(TAKEOVER_S,query,positions[:,i])) for i in range(2)],
        requested_takeover_speed_mps=speeds[len(targets)],
        delivered_takeover_speed_mps=float(np.linalg.norm((centers[index]-centers[index-1])/native_step)),
        continuous_reference_velocity_xy_mps=np.column_stack([np.interp(times,query,velocities[:,i]) for i in range(2)]).tolist(),
        reference_continuation='Smooth source-route continuation only; not an executed policy future',
        source_route_end_s=float(route_times[-1]),source_route_trimmed_before_reverse=route_count<len(times),
        delivered_speed_method='Completed history interval on the nominal native time grid; body centre',
        scope='Forward speed history; source initial pose/speed inherited; no lateral maneuver or route extrapolation')


def check_history_actors(frames, times, query, positions, angles):
    actors = [{obj['track']: obj for obj in row} for row in observed_objects_from_frames(frames)]
    for moment, position, yaw in zip(query, positions, angles):
        center = position + (EGO_FRONT - EGO_REAR)/2 * np.array([np.cos(yaw), np.sin(yaw)])
        ego = polygon(*center, yaw, EGO_FRONT + EGO_REAR, EGO_WIDTH)
        right = min(int(np.searchsorted(times, moment)), len(times)-1)
        left = max(0, right-1)
        weight = 0. if right == left else (moment-times[left]) / (times[right]-times[left])
        before, after = actors[left], actors[right]
        for track in before.keys() & after.keys():
            other = before[track]
            x, y = [other[key]*(1-weight) + after[track][key]*weight for key in ('x', 'y')]
            delta = after[track]['yaw'] - other['yaw']
            angle = other['yaw'] + weight*np.arctan2(np.sin(delta), np.cos(delta))
            if ego.intersects(polygon(x, y, angle, other['length'], other['width'])):
                raise ValueError(f'Ego history contacts observed actor {track} at {moment:.3f}s before policy takeover')


def reproject_frames(frames,xy,yaw):
    """Move ego sensors, preserving every actor in the established planar world frame."""
    times,old_xy,old_yaw=ego_path_from_frames(frames)
    actors=observed_objects_from_frames(frames);out=deepcopy(frames)
    first=np.asarray(frames[0]['ego2global']);yaw0=np.arctan2(first[1,0],first[0,0])
    rotation=np.array([[np.cos(yaw0),-np.sin(yaw0)],[np.sin(yaw0),np.cos(yaw0)]])
    for i,f in enumerate(out):
        pose=np.asarray(frames[i]['ego2global'],float).copy();delta=float(yaw[i]-old_yaw[i]);c,s=np.cos(delta),np.sin(delta)
        pose[:3,:3]=np.array([[c,-s,0],[s,c,0],[0,0,1]])@pose[:3,:3]
        pose[:2,3]=first[:2,3]+xy[i]@rotation.T
        f['ego2global']=pose.tolist();f['ego2global_translation']=pose[:3,3].tolist()
        if 'ego2global_rotation' in f:
            w,x,y,z=frames[i]['ego2global_rotation'];a,b=np.cos(delta/2),np.sin(delta/2)
            f['ego2global_rotation']=[a*w-b*z,a*x-b*y,a*y+b*x,a*z+b*w]
        lidar=np.asarray(f['lidar2ego'],float);f['lidar2global']=(pose@lidar).tolist()
        if 'lidar2global_translation' in f:f['lidar2global_translation']=(pose@lidar)[:3,3].tolist()
        if 'lidar2global_rotation' in f:
            w,x,y,z=frames[i]['lidar2global_rotation'];a,b=np.cos(delta/2),np.sin(delta/2)
            f['lidar2global_rotation']=[a*w-b*z,a*x-b*y,a*y+b*x,a*z+b*w]
        c,s=np.cos(yaw[i]),np.sin(yaw[i]);inverse=np.array([[c,s],[-s,c]])
        for j,actor in enumerate(actors[i]):
            box=np.asarray(f['anns']['gt_boxes'][j],float)
            local=inverse@(np.array([actor['x'],actor['y']])-xy[i])
            box[:2]=np.linalg.solve(lidar[:2,:2],local-lidar[:2,3]-lidar[:2,2]*box[2])
            box[6]=actor['yaw']-yaw[i]-np.arctan2(lidar[1,0],lidar[0,0])
            f['anns']['gt_boxes'][j]=box.tolist()
            if 'gt_velocity_3d' in f['anns']:
                velocity=np.asarray(f['anns']['gt_velocity_3d'][j],float)
                f['anns']['gt_velocity_3d'][j]=(np.array([[np.cos(delta),np.sin(delta),0],[-np.sin(delta),np.cos(delta),0],[0,0,1]])@velocity).tolist()
        if 'can_bus' in f:
            f['can_bus'][:3]=pose[:3,3].tolist()
            if 'ego2global_rotation' in f:f['can_bus'][3:7]=f['ego2global_rotation']
    return out


def verify(scene):
    scene=Path(scene)
    path=scene/'ego-history.json'
    if not path.is_file():return True
    data=json.loads(path.read_text());source=source_scene(scene)
    if hashlib.sha256((scene/'metadata/frames.json').read_bytes()).hexdigest()!=data['edited_frames_sha256']:
        raise ValueError('Prepared ego history was changed after generation')
    original=json.loads((source/'metadata/frames.json').read_text())
    edited=json.loads((scene/'metadata/frames.json').read_text())
    if data.get('mode') == 'endpoint_route':
        from ego_history_route import generate as generate_route
        xy, yaw, _, _ = generate_route(source, original, data['route'], data['acceptance_policy'], controls=data['controls'])
    else:
        xy,yaw,_=generate(original,data['targets'],data['acceptance_policy'])
    _,actual_xy,actual_yaw=ego_path_from_frames(edited)
    if not np.allclose(xy,actual_xy,atol=1e-7,rtol=0) or not np.allclose(yaw,actual_yaw,atol=1e-7,rtol=0):
        raise ValueError('Prepared ego history does not reproduce its motion request')
    return source.is_dir()


def clone(scene,output):
    """Import verified shared history into the new service output on resume."""
    scene=Path(scene)
    if not (scene/'ego-history.json').is_file():return scene
    verify(scene)
    output=Path(output)
    shutil.copytree(scene,output,symlinks=True)
    manifest=output/'ego-history.json';data=json.loads(manifest.read_text())
    data['prepared_scene']=str(output.resolve())
    manifest.write_text(json.dumps(data,indent=2))
    return output


def attach(scene,frozen):
    manifest=Path(scene)/'ego-history.json'
    if manifest.is_file():shutil.copyfile(manifest,Path(frozen)/'ego-history.json')


def prepare(scene,output,targets,acceptance,*,route=None):
    """Write a new scene; never modify source data or accumulated prefix edits."""
    scene=Path(scene).resolve()
    if targets is None and route is None:return scene
    source=source_scene(scene)
    if not targets and route is None:return source
    frames=json.loads((source/'metadata/frames.json').read_text())
    if route is not None:
        from ego_history_route import generate as generate_route
        xy, yaw, evidence, dense = generate_route(source, frames, route, acceptance)
    else:
        xy,yaw,evidence=generate(frames,targets,acceptance)
    from ego_road_checks import load_official_drivable,check_drivable_area_official
    times,_,_=ego_path_from_frames(frames)
    history_times=np.unique(np.r_[times[times<=TAKEOVER_S+.001],np.linspace(0.,TAKEOVER_S,61)])
    history_xy=np.column_stack([np.interp(history_times,times,xy[:,i]) for i in range(2)])
    history_yaw=np.interp(history_times,times,yaw)
    official = load_official_drivable(source/'metadata/map_drivable_official.json')
    road_args = dict(front=EGO_FRONT, rear=EGO_REAR, width=EGO_WIDTH, min_margin_m=acceptance['official_margin_m'])
    road = check_drivable_area_official(history_times,history_xy,history_yaw,official,**road_args)
    if not road['passed']:raise ValueError('Edited ego history leaves the drivable area')
    if route is not None:
        dense_road = check_drivable_area_official(*dense, official, **road_args)
        if not dense_road['passed']:
            raise ValueError('Dense ego route history leaves the drivable area')
        check_history_actors(frames, times, history_times, history_xy, history_yaw)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'metadata',output/'metadata')
    for name in ('source_asset','original_images'):
        if (source/name).exists():(output/name).symlink_to(source/name,target_is_directory=True)
    changed=reproject_frames(frames,xy,yaw)
    before=observed_objects_from_frames(frames);after=observed_objects_from_frames(changed)
    if any(abs(a[key]-b[key])>1e-7 for left,right in zip(before,after) for a,b in zip(left,right) for key in ('x','y','yaw')):
        raise ValueError('Ego history changed another actor in world coordinates')
    (output/'metadata/frames.json').write_text(json.dumps(changed))
    evidence.update(source_scene=str(source),source_frames_sha256=hashlib.sha256((source/'metadata/frames.json').read_bytes()).hexdigest(),
        prepared_scene=str(output.resolve()),
        acceptance_policy=deepcopy(acceptance),
        edited_frames_sha256=hashlib.sha256((output/'metadata/frames.json').read_bytes()).hexdigest(),
        original_navigation_ego2global=[f['ego2global'] for f in frames])
    (output/'ego-history.json').write_text(json.dumps(evidence,indent=2))
    return output
