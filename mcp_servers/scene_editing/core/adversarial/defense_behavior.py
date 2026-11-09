"""Shared progress/reserve/motion evidence for construction, FOCI and Agent choice.

Preferences guide generation, never replace unchanged road/contact/motion checks.
"""
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon

VERSION='defense_behavior.v1'


def preferences():
    return json.loads(Path(__file__).with_name('defense_behavior_preferences.json').read_text())


class DrivingObjective:
    def __init__(self,times,reference_xy,actors,target_track,effective):
        self.times=np.asarray(times,float);self.reference=np.asarray(reference_xy,float)
        self.boundary=float(effective['observation_end_s'])
        self.comfort=float(effective.get('comfort_reference_mps2',2.4))
        self.settings=preferences()
        segments=np.diff(self.reference,axis=0);length=np.linalg.norm(segments,axis=1)
        valid=length>1e-8
        self.origins=self.reference[:-1][valid];self.vectors=segments[valid]
        self.length=length[valid];self.arc=np.r_[0.,np.cumsum(length)][:-1][valid]
        if not len(self.length):raise ValueError('Reference route requires nonzero length')
        self.total=float(length.sum())
        self.reference_s=self.project(self.reference)[0]
        self.start_s=float(np.interp(self.boundary,self.times,self.reference_s))
        self.available=max(1e-6,self.total-self.start_s)
        self.mask=self.times>self.boundary
        self.actor_polygons=[]
        for frame in actors:
            a=next(a for a in frame if a['track']==target_track)
            self.actor_polygons.append(self.box(a['x'],a['y'],a['yaw'],a['length'],a['width']))
        # Pacifica dimensions and rear-axle convention shared with existing checks.
        from run_ego_reference import EGO_FRONT,EGO_REAR,EGO_WIDTH
        self.front,self.rear,self.width=EGO_FRONT,EGO_REAR,EGO_WIDTH
        self.grid=np.linspace(self.boundary,self.times[-1],max(3,int(np.ceil((self.times[-1]-self.boundary)/.1))+1))

    @staticmethod
    def box(x,y,yaw,length,width):
        points=np.array([[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2]])
        c,s=np.cos(yaw),np.sin(yaw)
        return Polygon(points@np.array([[c,s],[-s,c]])+[x,y])

    def project(self,xy):
        delta=np.asarray(xy,float)[:,None,:]-self.origins[None,:,:]
        u=np.clip(np.sum(delta*self.vectors,axis=2)/(self.length**2),0.,1.)
        nearest=self.origins[None,:,:]+u[:,:,None]*self.vectors[None,:,:]
        distances=np.sum((np.asarray(xy)[:,None,:]-nearest)**2,axis=2)
        idx=distances.argmin(axis=1);rows=np.arange(len(xy))
        return self.arc[idx]+u[rows,idx]*self.length[idx],np.sqrt(distances[rows,idx])

    def gaps(self,xy,yaw):
        shift=(self.front-self.rear)/2
        return np.array([self.box(p[0]+shift*np.cos(a),p[1]+shift*np.sin(a),a,
            self.front+self.rear,self.width).distance(target)
            for p,a,target in zip(xy,yaw,self.actor_polygons)])

    def evaluate(self,xy,yaw,clearance=None):
        xy=np.asarray(xy,float);yaw=np.asarray(yaw,float)
        route_s,offset=self.project(xy)
        start=float(np.interp(self.boundary,self.times,route_s))
        forward=route_s[-1]-start
        remaining=np.maximum(self.reference_s-route_s,0.)/self.available
        remaining_grid=np.interp(self.grid,self.times,remaining)
        progress_loss=float(.5*max(0.,1.-forward/self.available)+.5*np.trapz(remaining_grid,self.grid)/(self.grid[-1]-self.grid[0]))
        gap=self.gaps(xy,yaw) if clearance is None else np.asarray(clearance,float)
        # Rates computed on a uniform grid avoid micro-timestamp derivative spikes.
        gap_grid=np.interp(self.grid,self.times,gap)
        closing=np.maximum(-np.gradient(gap_grid,self.grid),0.)
        p=self.settings
        reserve=p['base_reserve_m']+p['geometry_allowance_m']+closing*p['closing_response_s']+closing**2/(2*self.comfort)
        shortfall=np.maximum(reserve-gap_grid,0.)/reserve
        reserve_cost=float(np.mean(shortfall**2))
        route_grid=np.interp(self.grid,self.times,route_s)
        route_speed=np.gradient(route_grid,self.grid)
        reverse=float(np.trapz(np.maximum(-route_speed,0.),self.grid))
        grid_xy=np.column_stack([np.interp(self.grid,self.times,xy[:,j]) for j in range(2)])
        velocity=np.gradient(grid_xy,self.grid,axis=0)
        acceleration=np.gradient(velocity,self.grid,axis=0)
        jerk=np.gradient(acceleration,self.grid,axis=0)
        acceleration_cost=float(np.mean(np.sum(acceleration**2,axis=1))/self.comfort**2)
        jerk_cost=float(np.mean(np.sum(jerk**2,axis=1))/self.comfort**2)
        path_cost=float(np.mean(offset[self.mask]**2)/self.width**2)
        terminal_closing=float(closing[-1]);terminal_cost=(terminal_closing/max(1.,self.comfort))**2
        costs=dict(progress_loss=progress_loss,reserve_shortfall=reserve_cost,acceleration=acceleration_cost,
            jerk=jerk_cost,reverse_distance=reverse/self.available,path_offset=path_cost,terminal_closing=terminal_cost)
        total=(p['progress_weight']*progress_loss+p['reserve_shortfall_weight']*reserve_cost+
            p['acceleration_weight']*acceleration_cost+p['jerk_weight']*jerk_cost+
            p['reverse_weight']*costs['reverse_distance']+p['path_offset_weight']*path_cost+
            p['terminal_closing_weight']*terminal_cost)
        critical=int(np.argmax(shortfall))
        return dict(version=VERSION,net_reference_progress_m=float(forward),reference_available_after_history_m=self.available,
            net_reference_progress_fraction=float(forward/self.available),reverse_reference_distance_m=reverse,
            mean_progress_loss=progress_loss,reserve_shortfall_cost=reserve_cost,
            max_reserve_shortfall_m=float(np.maximum(reserve-gap_grid,0.).max()),
            worst_reserve_time_s=float(self.grid[critical]),min_reserve_surplus_m=float((gap_grid-reserve).min()),
            terminal_closing_speed_mps=terminal_closing,actual_acceleration_cost=acceleration_cost,actual_jerk_cost=jerk_cost,
            acceleration_rms_mps2=float(np.sqrt(acceleration_cost)*self.comfort),
            jerk_rms_mps3=float(np.sqrt(jerk_cost)*self.comfort),
            path_offset_rms_m=float(np.sqrt(np.mean(offset[self.mask]**2))),
            optimization_cost=float(total),cost_components=costs,
            scope='Net route projection after observation; no credit for lateral detours/backtracking. Reserve shortfall saturates at zero: no reward for excess gap. Uniform-grid derivative diagnostics; native checks/analytic bounds remain authoritative. Research preferences, not calibrated robustness proof.')


def adaptive_spec(ledger, decisions, boundary, end):
    """Refine a checked family from its progress/reserve feedback, retaining failures."""
    usable=[r for r in ledger if r.get('operation',{} ) and r['operation'].get('edit_mode')=='brake_profile'
            and r.get('behavior_metrics')]
    if not usable:return None
    passed=[r for r in usable if r['reference_screen_passed']]
    anchor=min(passed or usable,key=lambda r:r['behavior_metrics']['optimization_cost'])
    op=anchor['operation'];start=float(op['start_time_s'])
    fraction=1./(2**(1+sum(bool(r.get('adaptive_proposal')) for r in ledger)))
    # Start-time changes are tied to the checked scene span, with bounded refinement.
    step=max(.02,(end-boundary)*fraction/4.)
    behavior=anchor['behavior_metrics']
    later=anchor['reference_screen_passed'] and behavior['max_reserve_shortfall_m']<=1e-6
    new_start=min(end-.01,start+step) if later else max(boundary,start-step)
    spec={k:op[k] for k in ('deceleration_mps2','target_speed_mps','lateral_offset_m') if k in op}
    proposed=('brake_profile',new_start,spec)
    signature=lambda item:json.dumps(item,sort_keys=True)
    if signature(proposed) in {signature(x) for x in decisions}:return None
    return proposed


AGENT_FIELDS=('version','net_reference_progress_m','reference_available_after_history_m',
    'net_reference_progress_fraction','reverse_reference_distance_m','max_reserve_shortfall_m',
    'worst_reserve_time_s','min_reserve_surplus_m','terminal_closing_speed_mps',
    'acceleration_rms_mps2','jerk_rms_mps3','path_offset_rms_m','scope')


def agent_evidence(metrics):
    return {key:(round(value,3) if isinstance(value,float) else value)
            for key,value in metrics.items() if key in AGENT_FIELDS}


def candidate_differences(catalog):
    """Compute pair differences, without sorting/recommending the candidate list."""
    rows=[r for r in catalog if r.get('status')=='reference_screen_passed' and r.get('behavior_metrics')]
    result=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:i+2]:
            am,bm=a['behavior_metrics'],b['behavior_metrics']
            result.append(dict(candidate_a=a['candidate_id'],candidate_b=b['candidate_id'],
                b_minus_a_net_reference_progress_m=round(bm['net_reference_progress_m']-am['net_reference_progress_m'],3),
                b_minus_a_min_target_clearance_m=round(b['min_target_clearance_m']-a['min_target_clearance_m'],3),
                b_minus_a_peak_acceleration_mps2=round(b['max_acceleration_mps2']-a['max_acceleration_mps2'],3),
                a_reserve_shortfall_m=am['max_reserve_shortfall_m'],b_reserve_shortfall_m=bm['max_reserve_shortfall_m']))
            if len(result)>=10:return result
    return result
