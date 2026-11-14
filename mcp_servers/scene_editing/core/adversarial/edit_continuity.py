"""One parent motion representation and explicit body-heading derivatives.

Interpolation defines missing sampled motion; delivered history/check grids remain
unchanged. A centre trajectory and a rear-axle trajectory have different headings.
"""
import numpy as np
from scipy.interpolate import BPoly, CubicSpline, CubicHermiteSpline, PPoly
from scipy.integrate import solve_ivp


def _concatenate_curves(parts):
    """Join polynomial pieces without changing their degrees or coordinates."""
    degree=max(part.c.shape[0] for part in parts)
    coefficients=[]
    for part in parts:
        pad=[(degree-part.c.shape[0],0)]+[(0,0)]*(part.c.ndim-1)
        coefficients.append(np.pad(part.c,pad))
    return PPoly(np.concatenate(coefficients,axis=1),
                 np.concatenate([parts[0].x]+[part.x[1:] for part in parts[1:]]))


def restrict_curve(curve,start):
    """Retain the exact polynomial from an arbitrary takeover time onward."""
    i=max(0,min(len(curve.x)-2,int(np.searchsorted(curve.x,start,side='right'))-1))
    shift=float(start-curve.x[i]);first=np.zeros_like(curve.c[:,i])
    for index in np.ndindex(curve.c.shape[2:]):
        key=(slice(None),i)+index
        polynomial=np.polynomial.Polynomial(curve.c[key][::-1])
        shifted=polynomial(np.polynomial.Polynomial([shift,1.])).coef
        first[(slice(-len(shifted),None),)+index]=shifted[::-1]
    return PPoly(np.concatenate([first[:,None],curve.c[:,i+1:]],axis=1),
                 np.r_[float(start),curve.x[i+1:]])


def _smooth_future_connection(q,points,angles,t,x,future,future_yaw):
    """Fit the missing motion over a real sample interval, not a micro seam.

    Native delivery stays untouched. Connections must respect the existing
    1cm reconstruction bound and preserve the defined future through C2.
    """
    from ego_trajectory_edit import curve_evidence
    join=float(future.x[0]);source=CubicSpline(q[:-1],points[:-1])
    source_yaw=CubicSpline(q[:-1],angles[:-1])
    spacing=np.diff(t);spacing=spacing[spacing>=.001]
    interval=float(np.median(spacing)) if len(spacing) else join-q[0]
    choices=[]
    for left in source.x[source.x<=join-interval+1e-8][::-1]:
        bridge=PPoly.from_bernstein_basis(BPoly.from_derivatives([left,join],
            [[source(left),source(left,1),source(left,2)],
             [future(join),future(join,1),future(join,2)]]))
        mask=(t>=left)&(t<join)
        error=float(np.max(np.linalg.norm(bridge(t[mask])-x[mask],axis=1))) if np.any(mask) else 0.
        if error>.01:continue
        peak=curve_evidence(bridge,{},'Reconstruction connection extrema')['max_acceleration_mps2']
        choices.append((peak,error,float(left),bridge))
    if not choices:
        raise ValueError('state_incompatible: cannot connect sampled history to the defined future within the 0.01m reconstruction bound; choose a later supported takeover or a different parent')
    peak,error,left,bridge=min(choices,key=lambda item:(item[0],item[1],-item[2]))
    yaw_bridge=PPoly.from_bernstein_basis(BPoly.from_derivatives([left,join],
        [[source_yaw(left),source_yaw(left,1),source_yaw(left,2)],
         [future_yaw(join),future_yaw(join,1),future_yaw(join,2)]]))
    # source.x includes left; truncating its right end keeps every earlier piece.
    index=int(np.searchsorted(source.x,left))
    xy_parts=([PPoly(source.c[:,:index],source.x[:index+1])] if index else [])+[bridge]
    yaw_parts=([PPoly(source_yaw.c[:,:index],source_yaw.x[:index+1])] if index else [])+[yaw_bridge]
    return _concatenate_curves(xy_parts),_concatenate_curves(yaw_parts),dict(
        method='bounded_C2_sample_to_future',window_s=[left,join],
        maximum_sample_adjustment_m=error,peak_acceleration_mps2=peak,
        position_bound_m=.01)


def _complete_defined_parent(times,lookup,parent_id,curve):
    """Add missing legacy yaw and the explicitly declared terminal hold."""
    cont=curve['continuous_state'];knots=np.asarray(cont['times_s'])
    p=PPoly(np.asarray(cont['xy_coefficients']),knots)
    hold=curve.get('hold_starts_at_s');changed=False
    if 'yaw_coefficients' in cont:
        a=PPoly(np.asarray(cont['yaw_coefficients']),knots)
    else:
        _,angles=lookup(parent_id)
        sampled=CubicSpline(np.asarray(times),np.unwrap(angles))
        # Use one polynomial representation/grid for both position and yaw.
        rates=sampled(knots,1);values=sampled(knots)
        if hold is not None and hold<=knots[-1]+1e-8:rates[-1]=0.
        a=CubicHermiteSpline(knots,values,rates);changed=True
    if knots[-1]<times[-1]-1e-8:
        if hold is None or hold>knots[-1]+1e-8:
            raise ValueError('state_incompatible: parent curve ends before scene end without a declared hold')
        last=float(knots[-1]);position=p(last);angle=float(a(last))
        p=_concatenate_curves([p,PPoly(position[None,None,:],[last,float(times[-1])])])
        a=_concatenate_curves([a,PPoly(np.array([[angle]]),[last,float(times[-1])])]);changed=True
    if not changed:return curve
    result=dict(curve)
    result['continuous_state']=dict(cont,times_s=p.x.tolist(),xy_coefficients=p.c.tolist(),yaw_coefficients=a.c.tolist())
    return result


def _limit_terminal_overshoot(curve,times,points):
    """Keep a sampled stop forward instead of inventing reverse between samples.

    Only limit the final moving knot's derivative when its last cubic piece
    reverses along the final chord. Native positions and the zero-speed stop
    stay exact; the adjacent Hermite pieces remain C1.
    """
    delta=points[-1]-points[-2];length=float(np.linalg.norm(delta))
    if length<1e-10:return curve,False
    projected=np.polynomial.Polynomial((curve.c[:,-1]@(delta/length))[::-1]).deriv()
    duration=float(times[-1]-times[-2])
    query=[0.,duration]+[float(r.real) for r in projected.deriv().roots() if abs(r.imag)<1e-8 and 0<r.real<duration]
    if min(projected(np.asarray(query)))>=-1e-8:return curve,False
    velocity=curve(times,1);velocity[-1]=0.
    slope=delta/duration
    for axis in range(2):
        velocity[-2,axis]=0. if abs(slope[axis])<1e-12 else np.clip(velocity[-2,axis]/slope[axis],0.,3.)*slope[axis]
    return CubicHermiteSpline(times,points,velocity),True


def parent_motion(times, lookup, parent_id, curve, boundary, start, context=None):
    from ego_speed_profile import continuous_parent
    curve=continuous_parent(times,lookup,parent_id,curve,boundary)
    if curve and curve.get('continuous_state') and start>=curve['continuous_state']['times_s'][0]:
        return _complete_defined_parent(times,lookup,parent_id,curve)
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
    connection=None;terminal_limited=False
    if future is not None:
        xy_curve,yaw_curve,connection=_smooth_future_connection(q,points,angles,t,x,future,future_yaw)
    else:
        xy_curve=CubicSpline(q[:end],points[:end],bc_type=bc)
        if stop is not None:xy_curve,terminal_limited=_limit_terminal_overshoot(xy_curve,q[:end],points[:end])
        yaw_curve=CubicSpline(q[:end],angles[:end],bc_type=('not-a-knot',(1,0.)) if stop is not None else 'not-a-knot')
    if future is not None:
        xy_curve=_concatenate_curves([xy_curve,future])
        if 'yaw_coefficients' in original:
            yaw_curve=_concatenate_curves([yaw_curve,future_yaw])
        else:
            tail=CubicSpline(future.x,future_yaw(future.x))
            yaw_curve=_concatenate_curves([yaw_curve,tail])
        if xy_curve.x[-1]<t[-1]:
            if curve.get('hold_starts_at_s') is None or curve['hold_starts_at_s']>xy_curve.x[-1]+1e-8:
                raise ValueError('state_incompatible: parent continuous curve ends before scene end without a declared hold')
            final=xy_curve.x[-1];position=xy_curve(final);angle=float(yaw_curve(final))
            xy_curve=_concatenate_curves([xy_curve,PPoly(position[None,None,:],[final,t[-1]])])
            yaw_curve=_concatenate_curves([yaw_curve,PPoly(np.array([[angle]]),[final,t[-1]])])
    elif stop is not None:
        xy_curve=PPoly(np.concatenate([xy_curve.c,np.array([[0.,0.],[0.,0.],[0.,0.],points[stop]])[:,None,:]],axis=1),np.r_[xy_curve.x,q[-1]])
        yaw_curve=PPoly(np.concatenate([yaw_curve.c,np.array([0.,0.,0.,angles[stop]])[:,None]],axis=1),np.r_[yaw_curve.x,q[-1]])
    error=float(np.max(np.linalg.norm(xy_curve(t)-x,axis=1)))
    if error>.01:
        raise ValueError(f'state_incompatible: parent interpolation needs {error:.6f}m sample adjustment, above the existing 0.01m reconstruction bound; choose an explicit continuous parent')
    result=dict(profile='sampled_parent_motion.v1',
        continuous_state=dict(version='continuous.v2',times_s=xy_curve.x.tolist(),xy_coefficients=xy_curve.c.tolist(),yaw_coefficients=yaw_curve.c.tolist()),
        parent_motion_source=dict(method='bounded_sample_interpolation',samples='attack_dense' if samples is not None else 'native',max_sample_error_m=error,position_bound_m=.01,screening_samples_removed=False))
    if terminal_limited:result['parent_motion_source']['terminal_overshoot_limited']=True
    if future is not None:
        result['parent_motion_source']['preserved_future_curve_from_s']=float(future.x[0])
        result['parent_motion_source']['connection']=connection
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
