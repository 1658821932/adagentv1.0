"""Timed lateral intent on a known route, with explicit speed/progress semantics."""
import numpy as np
from scipy.interpolate import BPoly,CubicHermiteSpline,PPoly
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from route_edit_contract import validate


def generate(times,lookup,args,boundary,*,parent_curve=None,acceptance=None,parent_context=None):
    from ego_speed_profile import continuous_parent
    from ego_speed_schedule import Route,acceleration_peak
    from ego_trajectory_edit import inherited_brake_state,curve_evidence
    validate({k:v for k,v in args.items() if k not in {'mode','candidate_id','generation_source','route_edit_version'}})
    parent_id=args['parent_candidate_id'];px,py=lookup(parent_id);start=float(args['start_time_s'])
    if not boundary<=start<times[-1]:
        from edit_feedback import EditFeasibilityError
        raise EditFeasibilityError('immutable_history' if start<boundary else 'target_unreachable','Route edit start is outside the editable horizon',requested_start_s=start,earliest_editable_time_s=float(boundary))
    from edit_continuity import parent_motion,state_at,body_heading
    parent_curve=parent_motion(times,lookup,parent_id,parent_curve,boundary,start,parent_context)
    state=state_at(times,px,py,start,parent_curve,parent_context)
    route=Route(times,lookup,parent_id,state,parent_curve,'parent')
    cont=(parent_curve or {}).get('continuous_state');pc=PPoly(np.asarray(cont['xy_coefficients']),np.asarray(cont['times_s'])) if cont else None
    native_v=np.diff(px,axis=0)/np.diff(times)[:,None];mid=(times[:-1]+times[1:])/2
    signed=np.sum(native_v*np.column_stack([np.cos((py[:-1]+py[1:])/2),np.sin((py[:-1]+py[1:])/2)]),axis=1)
    if np.any(signed[mid>=start]<-1e-5) or state['signed_speed_mps']<0:
        raise ValueError('route_unavailable: edit_route currently needs a forward parent; prepare a forward/stop speed profile before lateral editing')
    initial=float(np.linalg.norm(state['velocity_xy_mps']))
    def speed(time):
        if pc is not None and time>=pc.x[0]:return float(np.linalg.norm(pc(min(time,pc.x[-1]),1)))
        if pc is not None and start<pc.x[0]:return initial
        return float(np.interp(time,np.r_[start,mid[mid>start]],np.r_[initial,np.linalg.norm(native_v[mid>start],axis=1)]))
    arrival=start+args['transition_duration_s'];end=arrival
    knots=[start,arrival];offsets=[0.,float(args['left_offset_m'])]
    if args['end_behavior']=='return_to_route':
        hold=float(args.get('hold_duration_s',0))
        if hold>0:knots.append(arrival+hold);offsets.append(offsets[-1])
        end=arrival+hold+args['return_duration_s'];knots.append(end);offsets.append(0.)
    if end>times[-1]+1e-8:raise ValueError(f'target_unreachable: route transition ends at {end:.3f}s beyond scene end {times[-1]:.3f}s; shorten durations or start earlier')
    if knots[-1]<times[-1]:knots.append(float(times[-1]));offsets.append(offsets[-1])
    lateral=BPoly.from_derivatives(knots,[[v,0.,0.] for v in offsets])
    policy=args.get('speed_policy','preserve_speed')
    def geometry(parameter):
        if parameter<route.path.x[0]-1e-7 or parameter>route.path.x[-1]+1e-7:raise ValueError(f'route_exhausted: requested route parameter {parameter:.7f}, known range [{route.path.x[0]:.7f},{route.path.x[-1]:.7f}]; speed_policy={policy}. Use a smaller offset or an earlier return, or a parent with more known route. No extrapolation')
        parameter=float(np.clip(parameter,route.path.x[0],route.path.x[-1]));d=route.path(parameter,1);dd=route.path(parameter,2);norm=float(np.linalg.norm(d))
        if norm<1e-9:raise ValueError('route_unavailable: degenerate reference tangent')
        tangent=d/norm;normal=np.array([-tangent[1],tangent[0]])
        curvature=float((d[0]*dd[1]-d[1]*dd[0])/norm**3)
        return tangent,normal,curvature,norm
    def motion(time,parameter):
        tangent,normal,k,norm=geometry(parameter);offset=float(lateral(time));rate=float(lateral(time,1));v=speed(time)
        scale=1-k*offset
        if scale<=.1:raise ValueError('target_unreachable: requested offset folds the reference route; reduce offset or use a different route')
        if policy=='preserve_speed':
            if abs(rate)>v+1e-7:raise ValueError(f'target_unreachable: lateral rate {abs(rate):.3f}m/s exceeds available speed {v:.3f}m/s near {time:.3f}s; lengthen transition, reduce offset, or edit speed first')
            travel=np.sqrt(max(0.,v*v-rate*rate))/scale
        else:travel=v
        vel=scale*travel*tangent+rate*normal
        return travel/norm,vel,route.path(parameter)+offset*normal
    origin=float(np.interp(route.origin,route.distance,route.parameter))
    # Split at intent phases and parent motion knots, retaining exact waits/stops.
    parent_summary=(parent_curve or {}).get('speed_edit_summary') or {}
    motion_knots=[v['time_s'] for v in parent_summary.get('targets',[])]
    if pc is not None:motion_knots+=pc.x.tolist()
    if (parent_curve or {}).get('hold_starts_at_s') is not None:motion_knots.append(parent_curve['hold_starts_at_s'])
    # Parent polynomial nodes can differ from an exact requested phase only by
    # floating-point roundoff (e.g. 4.499999999999996 vs 4.5). Retain exact intent
    # times and avoid creating femtosecond interpolation pieces. This does not
    # remove any screening sample or lower the delivered curve's extrema.
    extra=[]
    for value in sorted(motion_knots):
        if not start<value<times[-1]:continue
        if min(abs(value-k) for k in knots)<=1e-9 or (extra and value-extra[-1]<=1e-9):continue
        extra.append(value)
    phases=np.unique(np.r_[knots,extra])
    solutions=[];current=origin
    for left,right in ([] if policy=='preserve_progress' else zip(phases[:-1],phases[1:])):
        sol=solve_ivp(lambda time,value:[motion(time,float(np.clip(value[0],route.path.x[0],route.path.x[-1])))[0]],(left,right),[current],rtol=1e-11,atol=1e-12,max_step=.025,dense_output=True)
        if not sol.success:raise ValueError('route_unavailable: route integration failed')
        solutions.append((left,right,sol));current=float(sol.y[0,-1])
    crossings=[]
    for left,right,sol in solutions:
        a=float(sol.sol(left)[0]);b=float(sol.sol(right)[0])
        for value in route.path.x[(route.path.x>a+1e-8)&(route.path.x<b-1e-8)]:
            crossings.append(brentq(lambda time:float(sol.sol(time)[0]-value),left,right,xtol=1e-11))
    from trajectory_sampling import construction_times
    query=construction_times(phases,crossings,start,float(times[-1]));parameters=np.zeros(len(query))
    if policy=='preserve_progress':
        parameters=np.array([np.interp(route.project(pc(float(t))),route.distance,route.parameter) for t in query])
    for left,right,sol in solutions:
        m=(query>=left)&(query<=right);parameters[m]=sol.sol(query[m])[0]
    positions=[];velocities=[]
    for time,param in zip(query,parameters):
        _,vel,pos=motion(float(time),float(param))
        if policy=='preserve_progress':
            tangent,normal,k,norm=geometry(float(param));delta=route.path(param)-pc(time)
            denominator=norm**2+float(delta@route.path(param,2))
            if denominator<=1e-10:raise ValueError('route_unavailable: parent projection has no unique forward progress')
            rate=float(pc(time,1)@route.path(param,1))/denominator
            vel=(1-k*float(lateral(time)))*norm*rate*tangent+float(lateral(time,1))*normal
        positions.append(pos);velocities.append(vel)
    positions=np.asarray(positions);velocities=np.asarray(velocities)
    positions[0]=state['xy_m'];velocities[0]=state['velocity_xy_mps']
    spline=CubicHermiteSpline(query,positions,velocities)
    yaw_curve,heading_evidence=body_heading(spline,query,state,parent_context)
    xy=np.asarray(px).copy();yaw=np.asarray(py).copy();m=times>start;xy[m]=spline(times[m]);yaw[m]=yaw_curve(times[m])
    evidence=curve_evidence(spline,state,'Exact extrema of delivered route-edit cubic; whole-body road/contact checks retain their existing scope.')
    evidence.update(profile='lateral_route_edit.v2',edit_contract_version='trajectory_edit.v2',body_heading=heading_evidence,parent_motion_source=parent_curve.get('parent_motion_source'),
        continuous_state=dict(version='continuous.v2',times_s=query.tolist(),xy_coefficients=spline.c.tolist(),yaw_coefficients=yaw_curve.c.tolist()),
        terminal_velocity_xy_mps=spline(times[-1],1).tolist(),terminal_body_yaw_rad=float(yaw_curve(times[-1])),
        acceleration_peak=acceleration_peak(spline))
    # A child speed edit receives the edited geometric route, never the old route.
    edited_route=Route(times,lambda _: (xy,yaw),parent_id,state,evidence,'parent')
    evidence['speed_route']=edited_route.evidence()
    achieved_speed=np.linalg.norm(spline(query,1),axis=1);reference_speed=np.array([speed(q) for q in query])
    parent_travel=float(np.trapz(reference_speed,query));actual_travel=float(np.interp(parameters[-1],route.parameter,route.distance)-route.origin)
    arrival_index=int(np.argmin(np.abs(query-arrival)));_,normal,_,_=geometry(parameters[arrival_index])
    actual_offset=float((spline(arrival)-route.path(parameters[arrival_index]))@normal)
    return_error=None
    if args['end_behavior']=='return_to_route':
        param=float(np.interp(end,query,parameters));_,normal,_,_=geometry(param)
        return_error=float(abs((spline(end)-route.path(param))@normal))
    evidence['route_edit_summary']=dict(preserved_until_s=start,left_offset_m=float(args['left_offset_m']),
        arrival_time_s=float(arrival),return_completed_at_s=float(end) if args['end_behavior']=='return_to_route' else None,
        end_behavior=args['end_behavior'],speed_policy=policy,target_offset_error_m=float(abs(actual_offset-args['left_offset_m'])),actual_arrival_offset_m=actual_offset,max_speed_change_at_construction_knots_mps=float(np.max(np.abs(achieved_speed-reference_speed))),
        reference_progress_difference_m=actual_travel-parent_travel,route_initialization=route.initialization,
        return_offset_error_m=return_error,actual_motion=dict(peak_acceleration=evidence['acceleration_peak']),
        scope='Offsets are relative to the parent forward route tangent. Return removes lateral offset; same-time parent position is not promised. Exact continuous bounds, sampled road/contact screening, no physical certificate.')
    return xy,yaw,evidence
