"""One parent motion representation and explicit body-heading derivatives.

Interpolation defines missing sampled motion; delivered history/check grids remain
unchanged. A centre trajectory and a rear-axle trajectory have different headings.
"""
import numpy as np
from scipy.interpolate import BPoly, CubicSpline, CubicHermiteSpline, PPoly
from scipy.integrate import solve_ivp


def parent_motion(times, lookup, parent_id, curve, boundary, start, context=None):
    from ego_speed_profile import continuous_parent
    curve=continuous_parent(times,lookup,parent_id,curve,boundary)
    if curve and curve.get('continuous_state') and start>=curve['continuous_state']['times_s'][0]:
        return curve
    context=context or {}
    samples=context.get('samples')
    if samples is None:
        x,a=lookup(parent_id);t=np.asarray(times,float)
    else:t,x,a=samples;t=np.asarray(t,float)
    x=np.asarray(x,float);a=np.unwrap(np.asarray(a,float))
    # This is construction, not screening decimation. Recheck the reconstructed
    # curve at EVERY original sample and retain the original delivery prefix.
    keep=[0]
    for i in range(1,len(t)-1):
        if t[i]-t[keep[-1]]>=.001 and abs(t[i]-start)>=.001:keep.append(i)
    if t[-1]-t[keep[-1]]<.001 and len(keep)>1:keep.pop()
    keep.append(len(t)-1)
    q=np.asarray(sorted(set([*t[keep],float(start)])))
    points=np.column_stack([np.interp(q,t,x[:,i]) for i in range(2)])
    angles=np.interp(q,t,a)
    future=None
    if curve and curve.get('continuous_state') and start<curve['continuous_state']['times_s'][0]:
        original=curve['continuous_state'];future=PPoly(np.asarray(original['xy_coefficients']),np.asarray(original['times_s']))
        join=float(future.x[0]);before=q<join
        q=np.r_[q[before],join];points=np.vstack([points[before],future(join)])
        future_yaw=PPoly(np.asarray(original['yaw_coefficients']),future.x) if 'yaw_coefficients' in original else CubicSpline(t,a)
        angles=np.r_[angles[before],future_yaw(join)]
    moving=np.flatnonzero(np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-8)
    stop=None
    if len(moving) and moving[-1]+1<len(q)-1:
        stop=int(moving[-1]+1)
    end=stop+1 if stop is not None else len(q)
    if end<2:raise ValueError('route_exhausted: sampled parent has no usable motion')
    bc=('not-a-knot',(1,future(future.x[0],1))) if future is not None else ('not-a-knot',(1,np.zeros(2))) if stop is not None else 'not-a-knot'
    xy_curve=CubicSpline(q[:end],points[:end],bc_type=bc)
    yaw_curve=CubicSpline(q[:end],angles[:end],bc_type=('not-a-knot',(1,float(future_yaw(future.x[0],1)))) if future is not None else ('not-a-knot',(1,0.)) if stop is not None else 'not-a-knot')
    if future is not None:
        xy_curve=PPoly(np.concatenate([xy_curve.c,future.c],axis=1),np.r_[xy_curve.x,future.x[1:]])
        if 'yaw_coefficients' in original:
            yaw_curve=PPoly(np.concatenate([yaw_curve.c,future_yaw.c],axis=1),xy_curve.x)
        else:
            tail=CubicSpline(future.x,future_yaw(future.x))
            yaw_curve=PPoly(np.concatenate([yaw_curve.c,tail.c],axis=1),xy_curve.x)
        if xy_curve.x[-1]<t[-1]:
            if curve.get('hold_starts_at_s') is None or curve['hold_starts_at_s']>xy_curve.x[-1]+1e-8:
                raise ValueError('state_incompatible: parent continuous curve ends before scene end without a declared hold')
            final=xy_curve.x[-1];position=xy_curve(final);angle=float(yaw_curve(final))
            xy_curve=PPoly(np.concatenate([xy_curve.c,np.array([[0.,0.],[0.,0.],[0.,0.],position])[:,None,:]],axis=1),np.r_[xy_curve.x,t[-1]])
            yaw_curve=PPoly(np.concatenate([yaw_curve.c,np.array([0.,0.,0.,angle])[:,None]],axis=1),np.r_[yaw_curve.x,t[-1]])
    elif stop is not None:
        xy_curve=PPoly(np.concatenate([xy_curve.c,np.array([[0.,0.],[0.,0.],[0.,0.],points[stop]])[:,None,:]],axis=1),np.r_[xy_curve.x,q[-1]])
        yaw_curve=PPoly(np.concatenate([yaw_curve.c,np.array([0.,0.,0.,angles[stop]])[:,None]],axis=1),np.r_[yaw_curve.x,q[-1]])
    error=float(np.max(np.linalg.norm(xy_curve(t)-x,axis=1)))
    if error>.01:
        raise ValueError(f'state_incompatible: parent interpolation needs {error:.6f}m sample adjustment, above the existing 0.01m reconstruction bound; choose an explicit continuous parent')
    result=dict(profile='sampled_parent_motion.v1',
        continuous_state=dict(version='continuous.v2',times_s=xy_curve.x.tolist(),xy_coefficients=xy_curve.c.tolist(),yaw_coefficients=yaw_curve.c.tolist()),
        parent_motion_source=dict(method='bounded_sample_interpolation',samples='attack_dense' if samples is not None else 'native',max_sample_error_m=error,position_bound_m=.01,screening_samples_removed=False))
    if future is not None:
        result['parent_motion_source']['preserved_future_curve_from_s']=float(future.x[0])
        if curve.get('hold_starts_at_s') is not None:result['hold_starts_at_s']=curve['hold_starts_at_s']
    elif stop is not None:result['hold_starts_at_s']=float(q[stop])
    return result


def state_at(times,xy,yaw,time,curve,context=None):
    from ego_trajectory_edit import inherited_brake_state
    state=inherited_brake_state(times,xy,yaw,time,curve)
    cont=curve['continuous_state'];knots=np.asarray(cont['times_s']);p=PPoly(np.asarray(cont['xy_coefficients']),knots)
    if 'yaw_coefficients' in cont:
        a=PPoly(np.asarray(cont['yaw_coefficients']),knots);omega=float(a(time,1));alpha=float(a(time,2))
    else:
        a=CubicSpline(np.asarray(times),np.unwrap(yaw));omega=float(a(time,1));alpha=float(a(time,2))
    acceleration=p(time,2)
    if time>knots[-1] and curve.get('hold_starts_at_s') is not None:
        acceleration=np.zeros(2);omega=alpha=0.
    state.update(acceleration_xy_mps2=acceleration.tolist(),yaw_rate_radps=omega,yaw_acceleration_radps2=alpha,
        native_interval_s=float(np.median(np.diff(times))),
        reference_point=(context or {}).get('reference_point','ego_rear_axle'))
    samples=(context or {}).get('samples')
    if samples is not None and curve.get('profile')=='sampled_parent_motion.v1':
        raw_t,raw_xy,raw_yaw=map(np.asarray,samples)
        i=int(np.searchsorted(raw_t,time))
        if 0<i<len(raw_t) and abs(raw_t[i]-time)<1e-9 and time-raw_t[i-1]<.001:
            # At a near-coincident delivered boundary the actual immutable left
            # segment owns the state, rather than a two-sided interpolant.
            left=np.flatnonzero((raw_t<time-.001)&(raw_t>=time-.3))
            chosen=np.r_[left,i]
            if len(chosen)>=2:
                q=raw_t[chosen]-time;degree=min(2,len(q)-1)
                c=np.polynomial.polynomial.polyfit(q,raw_xy[chosen],degree)
                a=np.polynomial.polynomial.polyfit(q,np.unwrap(raw_yaw[chosen]),degree)
                velocity=c[1];angle=float(raw_yaw[i]);direction=np.array([np.cos(angle),np.sin(angle)])
                state.update(xy_m=raw_xy[i].tolist(),yaw_rad=angle,velocity_xy_mps=velocity.tolist(),
                    acceleration_xy_mps2=(2*c[2]).tolist() if degree>1 else [0.,0.],
                    signed_speed_mps=float(velocity@direction),lateral_speed_mps=float(velocity@np.array([-direction[1],direction[0]])),
                    yaw_rate_radps=float(a[1]),yaw_acceleration_radps2=float(2*a[2]) if degree>1 else 0.,
                    method='immutable_left_segment_state',source_interval_s=[float(raw_t[chosen[0]]),float(time)])
    return state


def body_heading(spline,query,state,context=None):
    """Keep the inherited body state; do not equate centre tangent with yaw."""
    context=context or {};query=np.asarray(query,float);start=float(query[0]);end=float(query[-1])
    offset=float(context.get('rear_axle_offset_m',0.))
    blend=min(float(state.get('native_interval_s',.5)),end-start)
    omega=float(state.get('yaw_rate_radps',0.));alpha=float(state.get('yaw_acceleration_radps2',0.))
    if offset>0:
        angle=state['yaw_rad'];normal=np.array([-np.sin(angle),np.cos(angle)])
        residual=float(spline(start,1)@normal-offset*omega)
        # Yaw rate (steering state) is inherited. A changed translational
        # acceleration must not be converted into an artificial lateral-slip
        # overshoot merely to retain the old yaw acceleration.
        correction=BPoly.from_derivatives([start,start+blend],[[residual,0.,0.],[0.,0.,0.]])
        def derivative(t,value):
            n=np.array([-np.sin(value[0]),np.cos(value[0])]);slip=float(correction(t)) if t<start+blend else 0.
            return [float((spline(t,1)@n-slip)/offset)]
        solved=solve_ivp(derivative,(start,end),[float(angle)],t_eval=query,rtol=1e-10,atol=1e-12,max_step=.01)
        if not solved.success:raise ValueError('state_incompatible: body-heading integration failed')
        values=solved.y[0];rates=np.array([derivative(float(t),[a])[0] for t,a in zip(query,values)])
        method='centre_velocity_with_rear_axle_constraint'
    else:
        velocity=spline(query,1);accel=spline(query,2);speed=np.linalg.norm(velocity,axis=1);moving=speed>1e-7
        values=np.unwrap(np.arctan2(velocity[:,1],velocity[:,0]))
        # Body heading remains forward during declared reverse travel.
        for i in range(len(values)):
            if moving[i]:
                reference=state['yaw_rad'] if i==0 else values[i-1]
                values[i]+=round((reference-values[i])/np.pi)*np.pi
        if np.any(moving):values[~moving]=np.interp(query[~moving],query[moving],values[moving])
        else:values[:]=state['yaw_rad']
        rates=np.zeros(len(query));rates[moving]=(velocity[moving,0]*accel[moving,1]-velocity[moving,1]*accel[moving,0])/speed[moving]**2
        delta=float(state['yaw_rad']-values[0]);delta_rate=omega-float(rates[0])
        correction=BPoly.from_derivatives([start,start+blend],[[delta,delta_rate,0.],[0.,0.,0.]])
        mask=query<start+blend;values[mask]+=correction(query[mask]);rates[mask]+=correction(query[mask],1)
        method='rear_axle_tangent_with_inherited_body_state'
    values[0]=state['yaw_rad'];rates[0]=omega
    return CubicHermiteSpline(query,values,rates),dict(method=method,rear_axle_offset_m=offset,body_state_connection_s=blend,
        inherited_yaw_rate_radps=omega,inherited_yaw_acceleration_radps2=alpha,
        scope='Kinematic reference reconstruction; unchanged motion checks still apply.')
