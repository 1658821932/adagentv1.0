"""Bounded initialization of a geometric route from a partially continuous legacy path."""
import numpy as np
from scipy.interpolate import BPoly, CubicSpline, PPoly

LEGACY_POSITION_TOLERANCE_M = .01


def initialize(times, lookup, parent_id, state, evidence, *, inherit_takeover_tangent=False):
    """Keep defined future geometry and join its derivatives over the missing interval.

    A CSV does not define curvature. Its editable prefix may move by at most 1cm
    during initialization; that error is reported. Exact takeover and future stop
    remain constraints, and this never edits the delivered observation history.
    """
    from ego_trajectory_edit import start_state
    times=np.asarray(times);xy,yaw=lookup(parent_id)
    continuous=evidence['continuous_state']
    curve=PPoly(np.asarray(continuous['xy_coefficients']),np.asarray(continuous['times_s']))
    join=float(curve.x[0]);stop=evidence.get('hold_starts_at_s')
    end=float(stop if stop is not None else curve.x[-1])
    takeover=float(state['time_s'])
    bridge_start=takeover if takeover<join else join
    initial=state if takeover<join else start_state(times,xy,yaw,bridge_start)
    bridge=None
    if bridge_start<join-1e-8:
        bridge=BPoly.from_derivatives([bridge_start,join],
            [[initial['xy_m'],initial['velocity_xy_mps'],[0.,0.]],
             [curve(join),curve(join,1),curve(join,2)]])
    prefix_times=np.r_[times[times<bridge_start],bridge_start]
    prefix_xy=np.vstack([np.asarray(xy)[times<bridge_start],initial['xy_m']])
    prefix=None
    if len(prefix_times)>1:
        prefix=CubicSpline(prefix_times,prefix_xy,bc_type=('natural',(1,np.asarray(initial['velocity_xy_mps']))))
    def values(query,derivative=0):
        query=np.asarray(query);answer=curve(np.clip(query,join,end),derivative)
        mask=query<bridge_start
        if np.any(mask):answer[mask]=prefix(query[mask],derivative)
        mask=(query>=bridge_start)&(query<join)
        if np.any(mask):answer[mask]=bridge(query[mask],derivative)
        return answer
    native=times[(times>=bridge_start)&(times<join)]
    error=float(np.max(np.linalg.norm(values(native)-np.stack([np.interp(native,times,np.asarray(xy)[:,i]) for i in range(2)],axis=1),axis=1))) if len(native) else 0.
    if error>LEGACY_POSITION_TOLERANCE_M:
        raise ValueError(f'route_unavailable: legacy connection needs {error:.4f}m position adjustment, exceeding initialization tolerance {LEGACY_POSITION_TOLERANCE_M}m; request an explicit spatial connection or use a saved route')
    # Sample a single C2 source on both sides, then use its velocity directions.
    # Dense source samples are decimated in distance to avoid tiny stop knots.
    lower=float(times[0]) if takeover<join else join
    query=np.unique(np.r_[np.arange(lower,end,.005),bridge_start,join,min(takeover,end),end])
    query=query[(query>=times[0])&(query<=end)]
    positions=values(query);velocity=values(query,1)
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
    keep=[0]
    forced={int(np.argmin(np.abs(query-v))) for v in [bridge_start,join,min(takeover,end),end]}
    for i in range(1,len(query)):
        if i in forced or lengths[i]-lengths[keep[-1]]>=.05:
            if lengths[i]-lengths[keep[-1]]<.005 and keep[-1] not in forced and len(keep)>1:keep.pop()
            if lengths[i]-lengths[keep[-1]]>1e-8:keep.append(i)
    if len(keep)<2:raise ValueError('route_exhausted: parent has no usable continuous route')
    selected=velocity[keep];norm=np.linalg.norm(selected,axis=1)
    tangent=np.zeros_like(selected);moving=norm>1e-8;tangent[moving]=selected[moving]/norm[moving,None]
    for i in np.flatnonzero(~moving):
        before=max(0,i-1);after=min(len(keep)-1,i+1)
        delta=positions[keep[after]]-positions[keep[before]]
        if np.linalg.norm(delta)<1e-10:raise ValueError('route_unavailable: no direction at a stationary endpoint')
        tangent[i]=delta/np.linalg.norm(delta)
    coordinate=lengths[keep];points=positions[keep]
    # Use C2 geometry: normal offsets differentiate the route tangent, so a C1
    # Hermite path would turn curvature jumps into offset-velocity jumps.
    pivot=int(np.argmin(np.abs(query[keep]-min(takeover,end))))
    if inherit_takeover_tangent:
        inherited=np.asarray(state['velocity_xy_mps']);norm=float(np.linalg.norm(inherited))
        if norm>1e-8:tangent[pivot]=inherited/norm*(1 if state['signed_speed_mps']>=0 else -1)
    parts=[]
    if pivot>0:parts.append(CubicSpline(coordinate[:pivot+1],points[:pivot+1],bc_type=((1,tangent[0]),(1,tangent[pivot]))))
    if pivot<len(keep)-1:parts.append(CubicSpline(coordinate[pivot:],points[pivot:],bc_type=((1,tangent[pivot]),(1,tangent[-1]))))
    path=PPoly(np.concatenate([p.c for p in parts],axis=1),np.concatenate([parts[0].x]+[p.x[1:] for p in parts[1:]]))
    info=dict(method='bounded_C2_legacy_connection',reconstructed=True,
        source='parent_continuous_positions_and_derivatives',connection_window_s=[bridge_start,join],
        max_source_position_adjustment_m=error,position_tolerance_m=LEGACY_POSITION_TOLERANCE_M,
        start_acceleration_assumption='zero in the undefined CSV connection',
        scope='Legacy CSV geometry is reconstructed within the reported tolerance; takeover pose/velocity and defined future positions are retained. No physical execution certificate.')
    return path,info
