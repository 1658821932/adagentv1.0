"""Cheap, bounded attack hypotheses in the editor's fixed scene coordinates.

These estimates help choose an experiment. They never establish VAD behavior,
expert feasibility, visibility, lane identity or acceptance of a candidate.
"""
from pathlib import Path
import math
import numpy as np
import acceptance_policy as policy
from run_ego_reference import EGO_FRONT,EGO_REAR,EGO_WIDTH,polygon


def initial_road_failure(times, xy, yaw, size, road, margin):
    """Describe a fixed-prefix road rejection requiring only the vehicle center
    to lie inside the official drivable area."""
    from ego_road_checks import check_drivable_area_official

    report = check_drivable_area_official(times, xy, yaw, road,
        front=0.0, rear=0.0, width=0.0, min_margin_m=0.0)
    if report['passed']:
        return None
    violations = report['violations']
    outside = [row for row in violations if row['reason'] == 'not_contained']
    violation = (outside or violations)[0]
    worst = report['worst_corner_margin_m']
    if outside:
        message = 'A fixed-prefix body corner is outside or on the boundary of the official drivable area.'
        code = 'initial_footprint_not_contained'
    else:
        message = (f'All fixed-prefix corners are inside the road. Minimum margin is '
                   f'{worst:.4f}m; required margin is {margin:.4f}m.')
        code = 'initial_road_margin_below_minimum'
    return dict(status='rejected', code=code, error=message, failed_checks=['official_road'],
        measurements=dict(time_s=round(violation['time_s'], 6),
            worst_corner_margin_m=round(worst, 6), required_margin_m=margin))


def find_road_compatible_asset(assets, requested, times, xy, yaw, road, margin):
    """Keep the pose and choose the closest-sized catalog asset that fits the road."""
    alternatives = (asset for asset in assets if asset['asset_id'] != requested['asset_id'])
    ranked = sorted(alternatives, key=lambda asset: (
        sum((size - nominal) ** 2 for size, nominal in zip(asset['size_m'], requested['size_m'])),
        asset['asset_id'],
    ))
    return next((asset for asset in ranked if initial_road_failure(
        times, xy, yaw, asset['size_m'], road, margin) is None), None)


def _sample(times,values,query):
    values=np.asarray(values)
    if values.ndim==1:return np.interp(query,times,values)
    return np.column_stack([np.interp(query,times,values[:,axis]) for axis in range(values.shape[1])])


def conflict_context(times,ego,yaw,constraints):
    times=np.asarray(times);ego=np.asarray(ego)
    boundary=float(constraints.get('observation_end_s',1.5))
    start=max(float(times[0]),min(boundary,float(times[-1])))
    from ego_trajectory_edit import start_state
    state=start_state(times,ego,yaw,start)
    speed=float(np.linalg.norm(state['velocity_xy_mps']))
    effective=policy.resolve(constraints.get('acceptance_policy'))
    window=constraints.get('event_window_s',[1.5,7.5])
    queries=np.unique(np.clip(np.r_[start,np.linspace(max(start,window[0]),min(times[-1],window[1]),6)],times[0],times[-1]))
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(ego,axis=0),axis=1))]
    start_arc=float(np.interp(start,times,arc))
    rows=[]
    for t in queries:
        point=_sample(times,ego,[t])[0];angle=float(_sample(times,np.unwrap(yaw),[t])[0])
        rows.append(dict(time_s=round(float(t),3),rear_axle_xy_m=np.round(point,3).tolist(),
            heading_rad=round(angle,4),reference_route_distance_from_takeover_m=round(float(np.interp(t,times,arc))-start_arc,3),
            constant_takeover_speed_travel_m=round(speed*max(0.,float(t)-start),3)))
    return dict(coordinate_frame='initial_ego',ego_reference_point='rear_axle',
        actor_reference_point='body_center',takeover_time_s=start,takeover_speed_mps=round(speed,3),
        takeover_speed_source=dict(method=state.get('method'),source_interval_s=state.get('source_interval_s'),
            scope='Inherited reference-state estimate, not a VAD response prediction'),
        reference_samples=rows,straight_stop_estimates=[dict(deceleration_mps2=round(float(a),3),
            travel_m=round(speed**2/(2*a),3),duration_s=round(speed/a,3)) for a in
            dict.fromkeys([effective['comfort_reference_mps2'],effective['max_acceleration_mps2']]) if a>0],
        scope='Unranked route/time and zero-reaction straight-braking estimates; not visibility, VAD prediction or a checked defense. Curvature consumes planar acceleration. No pre-takeover contact is permitted.')


def reaction_screen(times,ego,yaw,target_xy,target_yaw,size,constraints):
    """Five cheap ego hypotheses on one known route, at <=0.1s spacing."""
    times=np.asarray(times,float);ego=np.asarray(ego,float);yaw=np.unwrap(yaw)
    effective=policy.resolve(constraints.get('acceptance_policy'))
    start=float(constraints.get('observation_end_s',1.5));window=constraints.get('event_window_s',[1.5,7.5])
    end=min(float(times[-1]),float(window[1]))
    if start>=end:return dict(status='unavailable',reason='No post-takeover horizon')
    q=np.unique(np.r_[np.arange(start,end,.1),start,end])
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(ego,axis=0),axis=1))]
    origin=float(np.interp(start,times,arc));route_left=arc[-1]-origin
    evidence=conflict_context(times,ego,yaw,constraints);speed=evidence['takeover_speed_mps']
    age=q-start
    keep=np.r_[True,np.diff(arc)>1e-8]
    def route(distance):
        requested=origin+distance
        return _sample(arc[keep],ego[keep],requested),_sample(arc[keep],yaw[keep],requested),bool(np.any(distance>route_left+1e-6))
    scenarios=[('recorded_reference',None,None),('slower_reference',None,None),
               ('constant_takeover_speed',0.,None),
               ('comfort_braking',effective['comfort_reference_mps2'],None),
               ('hard_braking',effective['max_acceleration_mps2'],None)]
    target=_sample(times,target_xy,q);angles=_sample(times,np.unwrap(target_yaw),q)
    target_polygons=[polygon(*point,angle,*size[:2]) for point,angle in zip(target,angles)]
    rows=[]
    for name,a,_ in scenarios:
        stop_time=None
        if name=='recorded_reference':positions=_sample(times,ego,q);headings=_sample(times,yaw,q);exhausted=False
        elif name=='slower_reference':
            # Delayed route progress is a hypothesis, not a claim about VAD's turn policy.
            phase=start+.6*age;positions=_sample(times,ego,phase);headings=_sample(times,yaw,phase);exhausted=False
        else:
            dt=age if not a else np.minimum(age,speed/a)
            distances=speed*dt-.5*a*dt**2
            positions,headings,exhausted=route(distances)
            if a:stop_time=start+speed/a
        gaps=[];contact=[]
        for t,p,h,obstacle in zip(q,positions,headings,target_polygons):
            center=p+(EGO_FRONT-EGO_REAR)/2*np.array([np.cos(h),np.sin(h)])
            body=polygon(*center,h,EGO_FRONT+EGO_REAR,EGO_WIDTH)
            gaps.append(float(body.distance(obstacle)))
            if body.intersects(obstacle):contact.append(float(t))
        i=int(np.argmin(gaps))
        rows.append(dict(hypothesis=name,min_body_clearance_m=round(gaps[i],3),closest_time_s=round(float(q[i]),3),
            first_sampled_contact_s=None if not contact else round(contact[0],3),
            requested_stop_time_s=None if stop_time is None else round(stop_time,3),
            requested_stop_rear_axle_xy_m=None if stop_time is None or stop_time>end else np.round(positions[-1],3).tolist(),
            route_exhausted=exhausted,coverage_status='truncated_at_known_route_end' if exhausted else 'within_known_route'))
    return dict(status='estimated',hypotheses=rows,ranked=False,
        scope='Cheap sampled hypotheses only; no road/motion/other-actor defense checks or policy inference. Slower path is time-scaled by0.6; route exhaustion is truncation, not a stopped-policy prediction. Braking assumes immediate constant tangential deceleration, ignores curvature and does not prove expert feasibility.')


def render_conflict(times,ego,yaw,actors,road,path,constraints):
    from decision_scene_view import render_overview
    path=Path(path)
    render_overview(times,ego,yaw,actors,road,path,event_window_s=constraints.get('event_window_s'),
        route_time_labels=True,spatial_hint=constraints.get('spatial_hint'))
    return path


def preview_options(options,times,ego,yaw,actors,road,assets,constraints,required_operation=None,required_track=None):
    if not isinstance(options,list) or not 1<=len(options)<=3:raise ValueError('Preview needs1..3 hypotheses')
    from ego_road_checks import check_drivable_area_official
    from initial_motion_precheck import initial_state_precheck
    from types import SimpleNamespace
    results=[];ids=set()
    for option in options:
        identifier=option.get('option_id')
        if not isinstance(identifier,str) or not identifier or len(identifier)>80 or identifier in ids:
            raise ValueError('Preview option IDs must be distinct nonempty strings up to80chars')
        ids.add(identifier)
        row=dict(option_id=identifier,operation=option.get('operation'))
        try:
            operation=option['operation']
            if required_operation and operation!=required_operation:raise ValueError('Explicit operation lock applies')
            if operation=='add_vehicle':
                if set(option)!={'option_id','operation','asset_id','x_m','y_m','yaw_rad','speed_mps'}:raise ValueError('Invalid added preview fields')
                if required_track is not None:raise ValueError('Explicit target lock applies')
                asset=next((a for a in assets if a['asset_id']==option['asset_id']),None)
                if asset is None:raise ValueError('Unknown vehicle asset')
                values=[option[k] for k in ('x_m','y_m','yaw_rad','speed_mps')]
                if any(type(v) not in (int,float) or not math.isfinite(v) for v in values) or not 0<=values[-1]<=12:
                    raise ValueError('Finite pose/speed0..12 required')
                if constraints.get('stationary_target') and values[-1]!=0:raise ValueError('Explicit stationary target applies')
                angle=values[2];xy=np.asarray(values[:2])+(np.asarray(times)-times[0])[:,None]*values[-1]*np.array([np.cos(angle),np.sin(angle)])
                angles=np.full(len(times),angle);size=asset['size_m'];track=None
                road_check=check_drivable_area_official(times,xy,angles,road,front=size[0]/2,rear=size[0]/2,
                    width=size[1],min_margin_m=constraints.get('official_margin_m',.1))
                row['road_passed']=road_check['passed']
            elif operation=='edit_existing':
                if set(option)!={'option_id','operation','track_id'}:raise ValueError('Invalid existing preview fields')
                track=option['track_id']
                if required_track and track!=required_track:raise ValueError('Explicit target lock applies')
                seq=[next((a for a in frame if a['track']==track and a.get('label')=='vehicle'),None) for frame in actors]
                if any(a is None for a in seq):raise ValueError('Existing track must be a fully observed vehicle')
                xy=np.array([[a['x'],a['y']] for a in seq]);angles=np.unwrap([a['yaw'] for a in seq])
                size=[seq[0]['length'],seq[0]['width'],seq[0].get('height',1.5)]
                row['inherited_state_supported']=initial_state_precheck(SimpleNamespace(target=dict(operation=operation,
                    track_id=track,size_m=size),times=np.asarray(times),seed_xy=xy,seed_yaw=angles,
                    edit_start_s=float(constraints.get('attack_prefix_lock_s',0.)),constraints=constraints)) is None
            else:raise ValueError('Unknown preview operation')
            initial_road = initial_road_failure(times[:1], xy[:1], angles[:1], size, road,
                constraints.get('official_margin_m', .1))
            row['initial_road_passed'] = initial_road is None
            if initial_road is not None:
                row['initial_road_rejection'] = initial_road
            boundary=float(constraints.get('observation_end_s',1.5))
            q=np.unique(np.r_[np.arange(times[0],min(boundary,times[-1]),.1),min(boundary,times[-1])])
            points=_sample(times,xy,q);headings=_sample(times,angles,q)
            reference=_sample(times,ego,q);reference_yaw=_sample(times,np.unwrap(yaw),q)
            prefix_contact=False;other_contact=False;min_gap=float('inf')
            from vehicle_trajectory import actor_observation_series,actor_pose_at_time
            series=actor_observation_series(times,actors,track)
            max_gap=1.1*float(np.median(np.diff(times)))
            for t,p,h,e,eh in zip(q,points,headings,reference,reference_yaw):
                body=polygon(*p,h,*size[:2]);center=e+(EGO_FRONT-EGO_REAR)/2*np.array([np.cos(eh),np.sin(eh)])
                own=polygon(*center,eh,EGO_FRONT+EGO_REAR,EGO_WIDTH)
                min_gap=min(min_gap,float(body.distance(own)));prefix_contact|=body.intersects(own)
                for _,observed,poses in series:
                    pose=actor_pose_at_time(observed,poses,t,max_gap,scene_times=times)
                    if pose is not None:other_contact|=body.intersects(polygon(*pose[:2],pose[2],pose[3],pose[4]))
            row.update(status='previewed',prefix_target_contact=bool(prefix_contact),prefix_other_contact=bool(other_contact),
                prefix_min_body_clearance_m=round(min_gap,3),seed_initial_body_xy_m=np.round(xy[0],3).tolist(),
                reactions=reaction_screen(times,ego,yaw,xy,angles,size,constraints))
        except (ValueError,KeyError,TypeError) as error:row.update(status='invalid',reason=str(error)[:250])
        results.append(row)
    return dict(status='previewed',options=results,ranked=False,candidate_created=False,
        scope='Initial seed hypotheses only. Added straight motion is not a final edited trajectory; existing future is editable. Full surface/motion/actor/road checks and exact-candidate VAD testing remain required. No hiding or visibility guarantee.')
