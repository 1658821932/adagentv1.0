"""Behavior-level speed editing; geometry/state arithmetic belongs to the backend."""
import copy
import numpy as np

REQUIRED = {'parent_candidate_id','start_time_s','goal','reason'}
OPTIONAL = {'reach_target_time_s','target_speed_mps','max_deceleration_mps2','path_source','braking_style','preview_only'}


def tool():
    return dict(type='function',function=dict(name='adjust_trajectory_speed',
        description=('Change speed on an existing route. Use goal=stop to stop and remain still, or reduce_speed with a positive target. '
            'Everything through start_time_s is preserved. Optionally give reach_target_time_s; the backend computes compatible braking. '
            'Default keeps the parent route; recorded_baseline allows further travel along the original road route with a checked connection. '
            'smooth_deceleration eases braking in/out; constant_deceleration uses steady braking. '
            'preview_only=true screens a proposed edit without creating a candidate or spending a candidate slot. Preview cannot be submitted. '
            'An infeasible result is local to this request. All road/contact/motion checks remain required.'),
        parameters=dict(type='object',properties=dict(
            parent_candidate_id=dict(type='string',description='Existing constructor, FOCI or edited candidate to improve.'),
            start_time_s=dict(type='number',description='Preserve the parent up to this time, then change its future speed. Must be after observation.'),
            goal=dict(type='string',enum=['stop','reduce_speed']),
            reason=dict(type='string',description='What to preserve, what to improve and which tradeoff to check.'),
            reach_target_time_s=dict(type='number',description='Optional absolute time to reach the target speed. Omit to prefer the comfort reference, fitted to remaining time/route within the hard cap.'),
            target_speed_mps=dict(type='number',description='Positive speed for reduce_speed; stop automatically uses zero.'),
            max_deceleration_mps2=dict(type='number',description='Optional peak tangential deceleration cap, no higher than the existing hard policy.'),
            path_source=dict(type='string',enum=['parent','recorded_baseline'],description='Default parent preserves its spatial route. Choose recorded_baseline when further original-route travel is intended. No extrapolation.'),
            braking_style=dict(type='string',enum=['smooth_deceleration','constant_deceleration'],description='Default smooth_deceleration ramps braking in/out; it needs more time/distance at the same peak deceleration.'),
            preview_only=dict(type='boolean',description='Default false. True returns feasibility and measurements without a new candidate. Still uses one model action.')),
            required=sorted(REQUIRED),additionalProperties=False)))


def validate(arguments):
    from ego_edit_contract import fields_match
    fields_match(arguments,REQUIRED,REQUIRED|OPTIONAL|{'edit_mode'},'adjust_trajectory_speed')
    if not isinstance(arguments['reason'],str) or not arguments['reason'].strip():
        raise ValueError('reason must explain the intended change')
    if arguments['goal'] not in ('stop','reduce_speed'):
        raise ValueError('goal must be stop or reduce_speed')
    if arguments.get('path_source','parent') not in ('parent','recorded_baseline'):
        raise ValueError('path_source must be parent or recorded_baseline')
    if arguments.get('braking_style','smooth_deceleration') not in ('smooth_deceleration','constant_deceleration'):
        raise ValueError('Unknown braking_style')
    if 'preview_only' in arguments and type(arguments['preview_only']) is not bool:
        raise ValueError('preview_only must be boolean')
    for key in ('start_time_s','reach_target_time_s','target_speed_mps','max_deceleration_mps2'):
        if key in arguments and (type(arguments[key]) not in (int,float) or not np.isfinite(arguments[key])):
            raise ValueError(key+' must be a finite number')
    target=arguments.get('target_speed_mps',0.)
    if (arguments['goal']=='stop' and target!=0) or (arguments['goal']=='reduce_speed' and target<=0):
        raise ValueError('stop uses target_speed_mps=0; reduce_speed requires a positive target_speed_mps')


def continuous_parent(times,lookup,parent_id,curve,boundary):
    """Rebuild old braking evidence before querying an arbitrary continuous state."""
    if not curve or curve.get('continuous_state') or curve.get('profile')!='constant_tangential_deceleration_then_hold_or_cruise':
        return curve
    from ego_trajectory_edit import brake_along_path
    parent_xy,parent_yaw=lookup(parent_id)
    path_id=curve['reference_path_candidate_id']
    path_xy,path_yaw=lookup(path_id)
    target=curve['target_speed_mps']
    op=dict(parent_candidate_id=parent_id,path_candidate_id=path_id,
        start_time_s=curve['actual_start_state']['time_s'],deceleration_mps2=curve['tangential_deceleration_mps2'],
        target_speed_mps=target,motion_intent='forward_stop' if target==0 else 'forward',
        end_behavior='hold' if target==0 else 'cruise',curve_state_version='continuous.v1',
        lateral_offset_m=curve.get('lateral_offset_m',0.))
    for key in ('path_geometry_version','lateral_duration_s','speed_law'):
        if key in curve:op[key]=curve[key]
    xy,yaw,rebuilt=brake_along_path(times,parent_xy,parent_yaw,path_xy,path_yaw,op,boundary,with_details=True,parent_curve=curve)
    mask=times>op['start_time_s']
    if not np.allclose(xy[mask],parent_xy[mask],atol=1e-8,rtol=0) or not np.allclose(yaw[mask],parent_yaw[mask],atol=1e-8,rtol=0):
        raise ValueError('Cannot reconstruct the parent continuous curve; choose a native sample boundary or a different parent')
    return rebuilt


def generate_speed_edit(times,lookup,arguments,boundary,*,parent_curve=None,acceptance=None):
    from ego_trajectory_edit import inherited_brake_state,brake_along_path,_endpoint
    import acceptance_policy as policy
    validate({k:v for k,v in arguments.items() if k not in {'mode','candidate_id','generation_source'}})
    effective=acceptance or policy.resolve()
    parent_id=arguments['parent_candidate_id']
    parent_xy,parent_yaw=lookup(parent_id)
    parent_curve=continuous_parent(times,lookup,parent_id,parent_curve,boundary)
    start=_endpoint(arguments['start_time_s'],times)
    if not boundary<=start<times[-1]:
        raise ValueError(f'start_time_s must be in [{boundary:.3f}, {times[-1]:.3f})s; observation history is fixed')
    state=inherited_brake_state(times,parent_xy,parent_yaw,start,parent_curve)
    speed=float(np.linalg.norm(state['velocity_xy_mps']))
    target=float(arguments.get('target_speed_mps',0.))
    smooth=arguments.get('braking_style','smooth_deceleration')=='smooth_deceleration'
    factor=1.5 if smooth else 1.
    cap=float(arguments.get('max_deceleration_mps2',effective['max_acceleration_mps2']))
    if not 0<cap<=effective['max_acceleration_mps2'] or target>=speed or target>effective['max_speed_mps']:
        raise ValueError('Target speed must be below inherited speed; deceleration cap must be positive and within the unchanged policy')
    explicit_deadline='reach_target_time_s' in arguments
    requested_end=arguments.get('reach_target_time_s')
    if requested_end is None:
        strength=min(cap,effective.get('comfort_reference_mps2',2.4))
        requested_end=start+factor*(speed-target)/strength
    if not explicit_deadline:requested_end=min(requested_end,float(times[-1]))
    end=_endpoint(requested_end,times)
    if not start<end<=times[-1]:
        minimum=factor*(speed-target)/(times[-1]-start)
        raise ValueError(f'target_unreachable: requested target time {end:.3f}s; scene ends {times[-1]:.3f}s. Peak tangential deceleration >= {minimum:.3f} m/s^2 is needed to finish by scene end; specify a feasible deadline/cap or start earlier')
    required=factor*(speed-target)/(end-start)
    if required>cap+1e-8:
        raise ValueError(f'target_unreachable: peak tangential deceleration {required:.3f} m/s^2 required, cap {cap:.3f}; earliest target time {start+factor*(speed-target)/cap:.3f}s')
    source=arguments.get('path_source','parent')
    origin=np.asarray(state['xy_m'])
    if source=='parent' and parent_curve and parent_curve.get('continuous_state') and start>=parent_curve['continuous_state']['times_s'][0]:
        last=float(parent_curve['continuous_state']['times_s'][-1])
        future=np.arange(start+.025,last,.025)
        future=np.unique(np.r_[future,last])
        states=[inherited_brake_state(times,parent_xy,parent_yaw,float(t),parent_curve) for t in future if t>start]
        route_xy=np.array([v['xy_m'] for v in states]);route_yaw=np.array([v['yaw_rad'] for v in states])
    else:
        ref_xy,ref_yaw=lookup(parent_id if source=='parent' else 'recorded_baseline')
        # Project onto the reference, then retain forward geometry only. Start is always
        # the actual inherited pose/velocity, never a snapped or teleported pose.
        delta=np.diff(ref_xy,axis=0);square=np.sum(delta**2,axis=1)
        fractions=(np.sum((origin-ref_xy[:-1])*delta,axis=1)/np.maximum(square,1e-20)).clip(0,1)
        nearest=int(np.argmin(np.linalg.norm(ref_xy[:-1]+fractions[:,None]*delta-origin,axis=1)))
        route_xy=ref_xy[nearest+1:];route_yaw=ref_yaw[nearest+1:]
    if len(route_xy):
        keep=np.linalg.norm(route_xy-origin,axis=1)>max(1e-8,speed*.001)
        route_xy,route_yaw=route_xy[keep],route_yaw[keep]
    if not len(route_xy):
        raise ValueError('route_exhausted: no forward parent path remains; use recorded_baseline only if continuing along the original route is intended')
    route_xy=np.vstack([origin,route_xy]);route_yaw=np.r_[state['yaw_rad'],route_yaw]
    if not explicit_deadline:
        # Fit the implicit comfort request to the available geometric path and scene
        # time. Explicit Agent deadlines are never moved. Use the same Hermite
        # route construction as braking and integrate its actual length.
        from scipy.interpolate import CubicHermiteSpline
        from scipy.integrate import quad
        lengths=np.linalg.norm(np.diff(route_xy,axis=0),axis=1)
        keep=np.r_[True,lengths>1e-8]
        geo_xy,geo_yaw=route_xy[keep],np.unwrap(route_yaw[keep])
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(geo_xy,axis=0),axis=1))]
        tangent=np.column_stack([np.cos(geo_yaw),np.sin(geo_yaw)])
        tangent[0]=np.asarray(state['velocity_xy_mps'])/speed
        selected=[0]
        for i in range(1,len(arc)):
            if arc[i]-arc[selected[-1]]>max(1e-8,speed*.001):selected.append(i)
        if len(selected)<2:raise ValueError('route_exhausted: insufficient forward route')
        geometry=CubicHermiteSpline(arc[selected],geo_xy[selected],tangent[selected])
        available=sum(quad(lambda value:float(np.linalg.norm(geometry(value,1))),left,right,
            epsabs=1e-10,epsrel=1e-9)[0] for left,right in zip(geometry.x[:-1],geometry.x[1:]))
        # Distance = target * horizon + 0.5 * (initial-target) * transition duration.
        max_duration=2*(available-target*(times[-1]-start))/(speed-target)
        if max_duration<=0:raise ValueError('route_exhausted: insufficient route for the requested terminal cruise')
        end=float(min(end,start+max_duration))
        required=factor*(speed-target)/(end-start)
        if required>cap+1e-8:
            raise ValueError(f'target_unreachable: remaining route needs peak tangential deceleration {required:.3f} m/s^2, cap {cap:.3f}; choose an earlier start or explicitly continue on recorded_baseline')
    resolved=dict(edit_mode='brake_profile',parent_candidate_id=parent_id,path_candidate_id=parent_id if source=='parent' else 'recorded_baseline',
        start_time_s=start,deceleration_mps2=cap,target_speed_mps=target,transition_time_s=end,
        motion_intent='forward_stop' if target==0 else 'forward',end_behavior='hold' if target==0 else 'cruise',
        reason=arguments['reason'],curve_state_version='continuous.v1',path_geometry_version='merge_micro_knots.v1')
    if smooth:resolved['speed_law']='smoothstep.v1'
    # A verified continuous state also owns the exact origin when the parent is a CSV.
    inherited=dict(actual_start_state=copy.deepcopy(state))
    xy,yaw,curve=brake_along_path(times,parent_xy,parent_yaw,route_xy,route_yaw,resolved,boundary,with_details=True,parent_curve=inherited)
    curve['resolved_speed_request']=resolved
    curve['speed_edit_summary']=dict(goal=arguments['goal'],start_time_s=start,initial_speed_mps=speed,
        target_time_s=end,target_speed_mps=target,peak_tangential_deceleration_mps2=required,
        braking_style=arguments.get('braking_style','smooth_deceleration'),path_source=source,
        preserved_until_s=start,deadline_source='explicit_agent_target' if explicit_deadline else 'comfort_fitted_to_available_time_and_route',scope='Generated and screened reference curve; not a physical execution guarantee')
    return xy,yaw,curve
