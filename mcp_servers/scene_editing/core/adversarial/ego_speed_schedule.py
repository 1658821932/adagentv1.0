"""One signed-speed contract for driving, holding, restarting and reversing on a known route."""
import copy
import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PPoly
from scipy.integrate import cumulative_trapezoid

from speed_profile_contract import tool, validate, validate_targets


def sampled_target_errors(times,xy,yaw,targets):
    """FOCI preference uses the same signed speed/absolute target time meanings.

    Without a continuous curve these are interval estimates, not exact endpoint derivatives.
    """
    velocity=np.diff(xy,axis=0)/np.diff(times)[:,None]
    heading=(np.unwrap(yaw[:-1])+np.unwrap(yaw[1:]))/2
    signed=np.sum(velocity*np.column_stack([np.cos(heading),np.sin(heading)]),axis=1)
    rows=[]
    for p in targets:
        i=max(0,min(len(signed)-1,int(np.searchsorted(times,p['time_s'],side='left'))-1))
        rows.append(dict(time_s=p['time_s'],target_speed_mps=p['speed_mps'],
                         sampled_speed_mps=float(signed[i]),error_mps=float(signed[i]-p['speed_mps'])))
    return dict(targets=rows,mean_squared_error_mps2=float(np.mean([r['error_mps']**2 for r in rows])),
        scope='Soft FOCI search target, completed sample-interval signed speed estimate. Exact timed states require edit_speed_profile and generated-curve checks.')


class Route:
    """Monotone geometric route coordinate, independent of driving direction/time."""
    def __init__(self,times,lookup,parent_id,state,parent_curve,source):
        origin=np.asarray(state['xy_m'])
        if source=='parent' and parent_curve and parent_curve.get('speed_route'):
            saved=parent_curve['speed_route']
            self.path=PPoly(np.array(saved['coefficients']),np.array(saved['knots_m']))
            self.initialization=dict(method='reuse_saved_spatial_route',reconstructed=False,version=saved.get('version','legacy'))
            self._make_length_table()
            # Exact route position is stored for arbitrary-time inheritance.
            self.origin=self.project(origin)
            position=self.path(float(np.interp(self.origin,self.distance,self.parameter)))
            mismatch=float(np.linalg.norm(position-origin))
            if mismatch>1e-5:
                raise ValueError(f'route_unavailable: inherited pose is {mismatch:.6f}m off the saved route; a spatial connection is needed, not a silent refit')
            return
        if source=='parent' and parent_curve and parent_curve.get('continuous_state'):
            from route_bootstrap import initialize
            self.path,self.initialization=initialize(times,lookup,parent_id,state,parent_curve,inherit_takeover_tangent=True)
            self._make_length_table();self.origin=self.project(origin)
            return
        ref_xy,ref_yaw=lookup(parent_id if source=='parent' else 'recorded_baseline')
        ref_xy=np.asarray(ref_xy);ref_yaw=np.unwrap(ref_yaw)
        length=np.linalg.norm(np.diff(ref_xy,axis=0),axis=1)
        keep=np.r_[True,length>1e-8];ref_xy,ref_yaw=ref_xy[keep],ref_yaw[keep]
        if len(ref_xy)<2:raise ValueError('route_exhausted: reference has no usable known length')
        # Existing reversing time series need a stored spatial route or another
        # forward reference. Do not fit a loop through a repeated reverse traversal.
        heading=(ref_yaw[:-1]+ref_yaw[1:])/2
        signed=np.sum(np.diff(ref_xy,axis=0)*np.column_stack([np.cos(heading),np.sin(heading)]),axis=1)
        if np.any(signed<-1e-6):
            raise ValueError('route_unavailable: reversing CSV has no unambiguous spatial route; use recorded_baseline or an existing generated route')
        delta=np.diff(ref_xy,axis=0);square=np.sum(delta**2,axis=1)
        fractions=np.clip(np.sum((origin-ref_xy[:-1])*delta,axis=1)/square,0,1)
        projections=ref_xy[:-1]+fractions[:,None]*delta
        j=int(np.argmin(np.linalg.norm(projections-origin,axis=1)))
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(ref_xy,axis=0),axis=1))]
        origin_arc=arc[j]+fractions[j]*(arc[j+1]-arc[j])
        # Connect to actual pose rather than snap it onto a reference. Remove tiny
        # adjacent geometric knots at takeover, keeping inherited position exact.
        merge=max(1e-7,np.linalg.norm(state['velocity_xy_mps'])*.001)
        left=arc<origin_arc-merge;right=arc>origin_arc+merge
        points=np.vstack([ref_xy[left],origin,ref_xy[right]])
        at=int(left.sum());spatial=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        if len(spatial)<2:raise ValueError('route_exhausted: no reference length around takeover')
        # Position geometry determines path derivatives. Body yaw is not a
        # geometric tangent (sideslip and reverse make them different quantities).
        velocity=np.asarray(state['velocity_xy_mps']);speed=np.linalg.norm(velocity)
        if speed>1e-8:
            tangent=velocity/speed*(1 if state['signed_speed_mps']>=0 else -1)
        else:
            # At rest inherit the route direction from nearby positions, not a
            # change of body orientation inferred from a zero velocity vector.
            before=max(0,at-1);after=min(len(points)-1,at+1)
            tangent=points[after]-points[before];tangent/=np.linalg.norm(tangent)
        parts=[]
        if at>0:
            parts.append(CubicSpline(spatial[:at+1],points[:at+1],bc_type=('natural',(1,tangent))))
        if at<len(points)-1:
            parts.append(CubicSpline(spatial[at:],points[at:],bc_type=((1,tangent),'natural')))
        self.path=PPoly(np.concatenate([p.c for p in parts],axis=1),
                        np.concatenate([parts[0].x]+[p.x[1:] for p in parts[1:]]))
        fit_error=float(np.max(np.linalg.norm(self.path(spatial)-points,axis=1)))
        self.initialization=dict(method='position_geometry_with_inherited_start_tangent',reconstructed=True,
            source='parent_continuous_positions' if source=='parent' and parent_curve and parent_curve.get('continuous_state') else 'sampled_reference_positions',
            max_source_knot_error_m=fit_error,scope='C2 within each side of takeover; exact inherited tangent at takeover. Sampled-only geometry is an interpolation, not an execution model.')
        self._make_length_table();self.origin=float(np.interp(spatial[at],self.parameter,self.distance))

    def _make_length_table(self):
        # Arc inversion is numerical; generated planar curve extrema are still exact.
        self.parameter=np.unique(np.r_[self.path.x,np.linspace(self.path.x[0],self.path.x[-1],max(2,int(np.ceil((self.path.x[-1]-self.path.x[0])/.01))+1))])
        self.distance=np.r_[0.,cumulative_trapezoid(np.linalg.norm(self.path(self.parameter,1),axis=1),self.parameter)]
        if self.distance[-1]<1e-8:raise ValueError('route_exhausted: degenerate reference geometry')

    def project(self,xy):
        points=self.path(self.parameter);j=int(np.argmin(np.linalg.norm(points-np.asarray(xy),axis=1)))
        from scipy.optimize import minimize_scalar
        lo=max(0,j-1);hi=min(len(points)-1,j+1)
        if hi==lo:return float(self.distance[j])
        sol=minimize_scalar(lambda t:float(np.sum((self.path(t)-xy)**2)),bounds=(self.parameter[lo],self.parameter[hi]),method='bounded',options={'xatol':1e-12})
        candidates=[float(sol.x),float(self.path.x[0]),float(self.path.x[-1])]
        candidates.extend(self.path.x[(self.path.x>=self.parameter[lo])&(self.path.x<=self.parameter[hi])].tolist())
        parameter=min(candidates,key=lambda value:float(np.sum((self.path(value)-xy)**2)))
        return float(np.interp(parameter,self.parameter,self.distance))

    def drive(self,query,law):
        from scipy.integrate import solve_ivp
        parameter_start=float(np.interp(self.origin,self.distance,self.parameter))
        def derivative(t,value):
            tangent=self.path(value[0],1);norm=float(np.linalg.norm(tangent))
            if norm<1e-8:raise ValueError('route_unavailable: degenerate path tangent')
            return [float(law(t))/norm]
        parameter=np.empty(len(query));current=parameter_start
        # Integrate each declared phase separately. A zero/zero interval is an
        # exact wait, not a numerical drift while an adaptive step crosses phases.
        for left,right in zip(law.x[:-1],law.x[1:]):
            mask=(query>=left-1e-10)&(query<=right+1e-10)
            samples=query[mask]
            if not len(samples):continue
            if max(abs(float(law(t))) for t in (left,(left+right)/2,right))<1e-14:
                parameter[mask]=current
                continue
            solution=solve_ivp(derivative,(float(left),float(right)),[current],t_eval=samples,
                rtol=1e-10,atol=1e-12,max_step=.025)
            if not solution.success:raise ValueError('route_unavailable: route integration failed')
            parameter[mask]=solution.y[0];current=float(solution.y[0,-1])
        if parameter.min()<self.path.x[0]-1e-7 or parameter.max()>self.path.x[-1]+1e-7:
            raise ValueError(f'route_exhausted: requested path parameter [{parameter.min():.3f},{parameter.max():.3f}]m; known range [{self.path.x[0]:.3f},{self.path.x[-1]:.3f}]m. Start earlier, reduce speed/duration, or explicitly choose a longer known route; no extrapolation')
        parameter=np.clip(parameter,self.path.x[0],self.path.x[-1])
        first=self.path(parameter,1);norm=np.linalg.norm(first,axis=1)
        if np.any(norm<1e-8):raise ValueError('route_unavailable: degenerate path tangent')
        tangent=first/norm[:,None]
        return self.path(parameter),tangent*np.asarray(law(query))[:,None],np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))

    def evidence(self):
        return dict(version='spatial_route.v3',knots_m=self.path.x.tolist(),coefficients=self.path.c.tolist(),scope='Known route only; no extrapolation or rear-space certification')


def generate(times,lookup,args,boundary,*,parent_curve=None,acceptance=None,parent_context=None):
    from ego_speed_profile import continuous_parent
    from ego_trajectory_edit import inherited_brake_state,_endpoint,curve_evidence
    import acceptance_policy
    clean={k:v for k,v in args.items() if k not in {'mode','candidate_id','generation_source','speed_schedule_version'}}
    validate(clean)
    effective=acceptance or acceptance_policy.resolve()
    parent_id=args['parent_candidate_id'];parent_xy,parent_yaw=lookup(parent_id)
    cap=float(args.get('max_acceleration_mps2',effective['max_acceleration_mps2']))
    if not 0<cap<=effective['max_acceleration_mps2']:
        raise ValueError('Acceleration cap must be positive and within unchanged planar policy')
    retime=args['operation']=='retime_braking'
    segment=None
    if retime:
        segment=(parent_curve or {}).get('braking_segment')
        if segment is None and parent_curve and parent_curve.get('profile') in ('constant_tangential_deceleration_then_hold_or_cruise','smooth_deceleration_then_hold_or_cruise') and parent_curve.get('target_speed_mps')==0:
            segment=dict(start_time_s=float(parent_curve['actual_start_state']['time_s']),stop_time_s=float(parent_curve['hold_starts_at_s']),
                stop_xy_m=parent_curve['transition_xy_m'],transition_style='smooth' if parent_curve['profile']=='smooth_deceleration_then_hold_or_cruise' else 'constant_acceleration')
        if segment is None:
            raise ValueError('retiming_unavailable: parent has no explicit reconstructable braking-to-stop segment. Use set_speed_targets with a stated start and timed states; no braking onset is invented for sampled FOCI')
        original_start=float(segment['start_time_s']);braking_start=original_start+args['braking_start_shift_s']
        start=min(original_start,braking_start)
    else:start=_endpoint(args['start_time_s'],times)
    style=args.get('transition_style',segment['transition_style'] if retime else 'smooth')
    style_source='explicit_request' if 'transition_style' in args else 'inherited_parent_braking' if retime else 'default_for_new_speed_targets'
    factor=1.5 if style=='smooth' else 1.
    if not boundary<=start<times[-1]:
        from edit_feedback import EditFeasibilityError
        raise EditFeasibilityError('immutable_history' if start<boundary else 'target_unreachable',
            'Edit start must lie at or after observation and before scene end',requested_start_s=float(start),earliest_editable_time_s=float(boundary),latest_editable_time_s=float(times[-1]))
    if retime and not start<=braking_start<times[-1]:raise ValueError('target_unreachable: shifted braking start is outside editable horizon')
    from edit_continuity import parent_motion,state_at,body_heading
    parent_curve=parent_motion(times,lookup,parent_id,parent_curve,boundary,start,parent_context)
    state=state_at(times,parent_xy,parent_yaw,start,parent_curve,parent_context)
    initial=float(np.sign(state['signed_speed_mps'])*np.linalg.norm(state['velocity_xy_mps']))
    if abs(state['signed_speed_mps'])<1e-9 and np.linalg.norm(state['velocity_xy_mps'])>1e-6:
        raise ValueError('state_incompatible: inherited motion is lateral; use a spatial route edit')
    route=Route(times,lookup,parent_id,state,parent_curve,args.get('path_source','parent'))
    if retime:
        if initial<=0:raise ValueError('retiming_unavailable: braking-to-stop retiming requires a forward inherited state')
        if args['preserve']=='stop_time':end=float(segment['stop_time_s'])
        else:
            destination=route.project(np.asarray(segment['stop_xy_m']))
            remaining=destination-route.origin-initial*(braking_start-start)
            end=braking_start+2*remaining/initial
        if not braking_start<end<=times[-1]+1e-8:raise ValueError('target_unreachable: shifted braking cannot retain the requested stopping target inside the scene')
        targets=([dict(time_s=float(braking_start),speed_mps=initial)] if braking_start>start+1e-8 else [])+[dict(time_s=float(end),speed_mps=0.)]
    else:targets=copy.deepcopy(args['targets'])
    for p in targets:p['time_s']=_endpoint(p['time_s'],times)
    validate_targets(targets,start=start,end=float(times[-1]),max_speed=effective['max_speed_mps'])
    knots=[float(start)]+[p['time_s'] for p in targets];speeds=[initial]+[p['speed_mps'] for p in targets]
    for i,(v,w) in enumerate(zip(speeds[:-1],speeds[1:])):
        if v*w<-1e-10:raise ValueError('direction_change_requires_stop: supply an explicit zero-speed target between forward and reverse')
        required=factor*abs(w-v)/(knots[i+1]-knots[i])
        if required>cap+1e-8:
            from edit_feedback import EditFeasibilityError
            raise EditFeasibilityError('target_unreachable',f'interval {knots[i]:.3f}–{knots[i+1]:.3f}s exceeds the acceleration cap',
                interval_s=[float(knots[i]),float(knots[i+1])],required_peak_tangential_acceleration_mps2=float(required),acceleration_cap_mps2=cap,
                minimum_transition_duration_s=float(factor*abs(w-v)/cap))
    if knots[-1]<times[-1]-1e-8:knots.append(float(times[-1]));speeds.append(speeds[-1])
    if style=='smooth':law=CubicHermiteSpline(knots,speeds,np.zeros(len(knots)))
    else:
        durations=np.diff(knots);coef=np.stack([np.diff(speeds)/durations,np.asarray(speeds[:-1])])
        law=PPoly(coef,np.asarray(knots))
    integral=law.antiderivative()
    # Uniform construction times can differ from an explicit target only by
    # floating-point rounding. Keep exact targets, remove derived neighbors;
    # do not build nearly zero-duration cubic pieces or relax motion checks.
    from scipy.optimize import brentq
    path_distances=np.interp(route.path.x,route.parameter,route.distance)
    crossing_times=[]
    for left,right in zip(law.x[:-1],law.x[1:]):
        a=float(route.origin+integral(left)-integral(start));b=float(route.origin+integral(right)-integral(start))
        if abs(b-a)<1e-10:continue
        for distance in path_distances[(path_distances>min(a,b)+1e-8)&(path_distances<max(a,b)-1e-8)]:
            crossing_times.append(brentq(lambda time:float(route.origin+integral(time)-integral(start)-distance),left,right,xtol=1e-12))
    from trajectory_sampling import construction_times
    query=construction_times(knots,crossing_times,start,float(times[-1]))
    progress=route.origin+integral(query)-integral(start)
    if progress.min()<-1e-6 or progress.max()>route.distance[-1]+1e-6:
        from edit_feedback import EditFeasibilityError
        raise EditFeasibilityError('route_exhausted','The speed targets exceed the known route; no extrapolation',
            requested_arc_range_m=[float(progress.min()),float(progress.max())],known_arc_range_m=[0.,float(route.distance[-1])],
            missing_route_length_m=float(max(0.,-progress.min(),progress.max()-route.distance[-1])))
    positions,velocities,headings=route.drive(query,law)
    positions[0]=state['xy_m'];velocities[0]=state['velocity_xy_mps'];headings[0]=state['yaw_rad']
    spline=CubicHermiteSpline(query,positions,velocities)
    yaw_curve,heading_evidence=body_heading(spline,query,state,parent_context)
    xy,yaw=parent_xy.copy(),parent_yaw.copy();mask=times>start
    xy[mask]=spline(times[mask]);yaw[mask]=yaw_curve(times[mask])
    evidence=curve_evidence(spline,state,'Exact planar delivered cubic speed/acceleration extrema; road/contact checks retain sampled scope; no physical execution')
    terminal=float(speeds[-1]);hold_start=None
    if terminal==0:
        k=len(speeds)-1
        while k>0 and speeds[k-1]==0:k-=1
        hold_start=float(knots[k])
    preservation=None
    if retime:
        error=(float(np.linalg.norm(xy[-1]-np.asarray(segment['stop_xy_m']))) if args['preserve']=='stop_position' else abs(hold_start-float(segment['stop_time_s'])))
        preservation=dict(preserve=args['preserve'],error=error,tolerance=1e-4 if args['preserve']=='stop_position' else 1e-8,units='m' if args['preserve']=='stop_position' else 's')
    evidence.update(profile='signed_speed_schedule.v4',edit_contract_version='trajectory_edit.v2',
        body_heading=heading_evidence,parent_motion_source=parent_curve.get('parent_motion_source'),
        continuous_state=dict(version='continuous.v2',times_s=spline.x.tolist(),xy_coefficients=spline.c.tolist(),yaw_coefficients=yaw_curve.c.tolist()),
        speed_route=route.evidence(),reference_path_candidate_id=parent_id if args.get('path_source','parent')=='parent' else 'recorded_baseline',
        terminal_velocity_xy_mps=spline(times[-1],1).tolist(),terminal_body_yaw_rad=float(yaw_curve(times[-1])),
        target_speed_mps=terminal,hold_starts_at_s=hold_start,
        speed_edit_summary=dict(operation=args['operation'],preserved_until_s=float(start),path_source=args.get('path_source','parent'),
            transition_style=style,transition_style_source=style_source,route_initialization=route.initialization,targets=targets,preservation_evidence=preservation,terminal_behavior='hold' if terminal==0 else 'maintain_signed_speed',
            retiming=(dict(original_braking_start_s=original_start,actual_braking_start_s=float(braking_start),preserve=args['preserve']) if retime else None),
            actual_start_speed_mps=initial,peak_tangential_acceleration_mps2=float(max(abs(law(t,1)) for t in np.r_[knots,((np.array(knots[:-1])+np.array(knots[1:]))/2)])),
            target_evidence=[dict(time_s=p['time_s'],requested_speed_mps=p['speed_mps'],actual_speed_mps=float(np.copysign(np.linalg.norm(spline(p['time_s'],1)),spline(p['time_s'],1)@np.array([np.cos(yaw_curve(p['time_s'])),np.sin(yaw_curve(p['time_s']))]))),
                actual_body_forward_speed_mps=float(spline(p['time_s'],1)@np.array([np.cos(yaw_curve(p['time_s'])),np.sin(yaw_curve(p['time_s']))]))) for p in targets],
            scope='Signed-speed states on a checked known route; sampled geometry does not certify rear coverage or physical execution'),
        resolved_targets=[dict(time_s=p['time_s'],xy_m=spline(p['time_s']).tolist(),speed_mps=float(law(p['time_s']))) for p in targets],
        direction_sequence=[float(v) for v in speeds],declared_reverse=any(v<0 for v in speeds))
    evidence['acceleration_peak']=acceleration_peak(spline)
    evidence['speed_edit_summary']['actual_motion']=dict(peak_acceleration=evidence['acceleration_peak'],hard_planar_acceleration_limit_mps2=effective['max_acceleration_mps2'])
    if retime or (len(targets)==1 and initial>0 and targets[0]['speed_mps']==0):
        evidence['braking_segment']=dict(start_time_s=float(braking_start) if retime else float(start),stop_time_s=float(hold_start),stop_xy_m=xy[-1].tolist(),transition_style=style)
    return xy,yaw,evidence


def acceleration_peak(spline):
    """Actual delivered cubic acceleration maxima occur at piece endpoints."""
    best=None
    for i,span in enumerate(np.diff(spline.x)):
        for elapsed in (0.,float(span)):
            c=spline.c[:,i]
            velocity=3*c[0]*elapsed**2+2*c[1]*elapsed+c[2]
            acceleration=6*c[0]*elapsed+2*c[1]
            magnitude=float(np.linalg.norm(acceleration))
            speed=float(np.linalg.norm(velocity))
            along=float(acceleration@velocity/speed) if speed>1e-8 else None
            row=dict(time_s=float(spline.x[i]+elapsed),interval_s=[float(spline.x[i]),float(spline.x[i+1])],
                magnitude_mps2=magnitude,tangential_mps2=along,
                normal_mps2=None if along is None else float(np.sqrt(max(0.,magnitude**2-along**2))),
                scope='Exact cubic endpoint acceleration, not tire/physical dynamics')
            if best is None or magnitude>best['magnitude_mps2']:best=row
    return best


def apply_schedule_intent(result,times,xy,yaw):
    curve=result['generation_curve'];summary=curve['speed_edit_summary']
    # Direction changes are declared by the signed-speed targets and continuous
    # law, not hidden behind a generic unrestricted maneuver.
    matched=all(abs(row['actual_speed_mps']-row['requested_speed_mps'])<=1e-6 for row in summary['target_evidence'])
    stopped=bool(np.linalg.norm(curve['terminal_velocity_xy_mps'])<=1e-6)
    hold=True
    if curve.get('hold_starts_at_s') is not None:
        from ego_trajectory_edit import intent_evidence
        hold=intent_evidence(times,xy,yaw,'unrestricted',summary['preserved_until_s'],'hold',curve['hold_starts_at_s'])['hold_matched']
    preservation=summary.get('preservation_evidence')
    preservation_matched=preservation is None or preservation['error']<=preservation['tolerance']
    passed=bool(matched and hold and preservation_matched)
    result['motion_intent']=dict(passed=passed,declared='signed_speed_targets',direction_sequence=curve['direction_sequence'],
        declared_reverse=curve['declared_reverse'],terminal_stopped=stopped,terminal_state_method='reconstructed_generation_curve',
        targets_matched=matched,preservation_matched=preservation_matched,hold_matched=hold,start_s=summary['preserved_until_s'],failure_kind=None if passed else 'intent_mismatch')
    for name in ('checks','full_window_checks','editing_checks'):
        if isinstance(result.get(name),dict):result[name]['motion_intent']=passed
    for name in ('coverage','full_window_coverage','editing_coverage'):
        if isinstance(result.get(name),dict):result[name]['motion_intent']='passed' if passed else 'failed'
    if not passed:result.update(status='reference_screen_failed',editing_status='reference_screen_failed',full_window_status='reference_screen_failed',submit_status='rejected')
    return result
