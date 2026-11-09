"""Freeze the executed attack scene on an explicit 20 Hz ego-editing time base."""
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


VERSION='frozen_executed_replay.v1'


@dataclass(frozen=True)
class FrozenAttackScene:
    source_run:Path
    times_s:np.ndarray
    source_times_s:np.ndarray
    frame_ids:np.ndarray
    attack_xy:np.ndarray
    attack_heading_rad:np.ndarray
    attack_speed_mps:np.ndarray
    attack_z_m:np.ndarray
    ego_xy:np.ndarray
    ego_heading_rad:np.ndarray
    ego_speed_mps:np.ndarray
    ego_z_m:np.ndarray
    metadata:dict

    def planning_view(self):
        indices=np.arange(0,len(self.times_s),2)
        if indices[-1]!=len(self.times_s)-1:
            raise ValueError('Canonical 20 Hz input must end on a 10 Hz planning sample')
        values={name:(getattr(self,name)[indices] if isinstance(getattr(self,name),np.ndarray) else getattr(self,name))
            for name in ['times_s','source_times_s','frame_ids','attack_xy','attack_heading_rad','attack_speed_mps','attack_z_m',
                'ego_xy','ego_heading_rad','ego_speed_mps','ego_z_m']}
        metadata=dict(self.metadata,sampling='10Hz planning view derived from every second canonical tick')
        return replace(self,metadata=metadata,**values)


def _rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def _array(rows,names,dtype=float):
    values=np.array([[row[name] for name in names] for row in rows],dtype=dtype)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite frozen scene input')
    return values


def _speed(times,xy):
    if len(times)<2:
        raise ValueError('At least two states are required')
    return np.linalg.norm(np.gradient(xy,times,axis=0),axis=1)


def _resample(source_times,times,values):
    values=np.asarray(values,dtype=float)
    if values.ndim==1:
        return np.interp(times,source_times,values)
    return np.column_stack([np.interp(times,source_times,values[:,axis]) for axis in range(values.shape[1])])


def load_frozen_attack_scene(source_run):
    """Load gate+ticks; source_time_s stays unchanged while execution time starts at zero."""
    source_run=Path(source_run).resolve()
    fixed=_rows(source_run/'carla/executed-fixed.csv')
    ticks=_rows(source_run/'carla/event-ticks.csv')
    reference=[r for r in _rows(source_run/'llm/reference.csv') if r['trajectory_type']=='ego_future']
    validation=json.loads((source_run/'carla/validation.json').read_text())
    if not fixed or not ticks or len(reference)!=len(fixed):
        raise ValueError('Incomplete frozen scene sources')
    if int(validation['event_log_rows'])!=len(ticks) or int(validation['executed_samples'])!=len(fixed):
        raise ValueError('Archived row counts differ from execution validation')
    for row,ref in zip(fixed,reference):
        if int(row['frame_id'])!=int(ref['frame_id']) or abs(float(row['timestep'])-float(ref['timestep']))>1e-9:
            raise ValueError('Ego reference and fixed execution grids differ')
    if any(int(row['track_id'])!=int(validation['target_track_id']) for row in fixed):
        raise ValueError('Frozen attack identity differs from validation')

    count=len(ticks)+1
    times=np.arange(count,dtype=float)*float(validation['physics_timestep_seconds'])
    if len(times)<3 or not np.isclose(times[1]-times[0],.05):
        raise ValueError('Expected canonical 20 Hz execution ticks')
    raw_times=np.array([0.]+[float(r['event_time_after_s']) for r in ticks])
    if np.any(np.diff(raw_times)<=0):
        raise ValueError('Source event ticks are not strictly increasing')
    attack=np.zeros((count,2));attack_heading=np.zeros(count);attack_speed=np.zeros(count);attack_z=np.zeros(count)
    ego=np.zeros((count,2));ego_heading=np.zeros(count);ego_z=np.zeros(count)
    attack[0]=[float(fixed[0]['x']),float(fixed[0]['y'])]
    attack_heading[0]=float(fixed[0]['heading']);attack_speed[0]=float(fixed[0]['speed_mps']);attack_z[0]=float(fixed[0]['world_z_m'])
    ego[0]=[float(reference[0]['x']),float(reference[0]['y'])]
    ego_heading[0]=float(reference[0]['heading']);ego_z[0]=float(fixed[0]['ego_world_z_m'])
    attack[1:]=_array(ticks,['target_x_m','target_y_m'])
    attack_heading[1:]=_array(ticks,['target_heading_rad'])[:,0]
    attack_speed[1:]=_array(ticks,['target_speed_mps'])[:,0]
    attack_z[1:]=_array(ticks,['target_world_z_m'])[:,0]
    ego[1:]=_array(ticks,['ego_actual_x_m','ego_actual_y_m'])
    # The archived post-tick ego heading is not a separate column. Ego was kinematic;
    # use its same-tick applied command and retain this provenance in the metadata.
    ego_heading[1:]=_array(ticks,['ego_command_heading_rad'])[:,0]
    ego_z[1:]=_array(ticks,['ego_world_z_m'])[:,0]
    attack=_resample(raw_times,times,attack)
    attack_heading=_resample(raw_times,times,np.unwrap(attack_heading))
    attack_speed=_resample(raw_times,times,attack_speed);attack_z=_resample(raw_times,times,attack_z)
    ego=_resample(raw_times,times,ego);ego_heading=_resample(raw_times,times,np.unwrap(ego_heading))
    ego_z=_resample(raw_times,times,ego_z);source_times=times.copy()
    ego_speed=_speed(times,ego)
    frame_ids=float(fixed[0]['frame_id'])+10*times
    arrays=[times,source_times,frame_ids,attack,attack_heading,attack_speed,attack_z,ego,ego_heading,ego_speed,ego_z]
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError('Non-finite canonical scene')
    geometry=validation['vehicle_geometry']
    fixed_xy=_array(fixed,['x','y'])
    crosscheck=float(np.linalg.norm(attack[::2]-fixed_xy,axis=1).max())
    if crosscheck>1e-3:
        raise ValueError('Canonical 10 Hz attack samples differ from archived fixed-frame cross-check')
    metadata=dict(version=VERSION,source_run=str(source_run),attack_object_id='obj_002',
        target_track_id=int(validation['target_track_id']),coordinate_frame='fixed_frame0; x forward, y left, z up',
        attack_mode='frozen_executed_replay',execution_time_rule='t=0 at archived gate; canonical samples query the raw source timeline every 0.05 s',
        source_time_rule='source_time_s is the raw-timeline query time; states are linearly interpolated between gate and event_time_after_s samples, preserving the archived first-tick phase',
        first_tick_phase_s=float(ticks[0]['event_time_after_s'])-float(validation['physics_timestep_seconds']),
        ego_heading_source='t0 archived ego_future; post-tick same-tick kinematic ApplyTransform command',
        ego_role='Archived kinematic baseline and planning initial path only; not a new physical ego result',
        attack_scope='Fixed exogenous replay, including archived post-contact response; not counterfactual attacker physics',
        geometry=geometry,baseline_actual_contact_time_s=validation.get('first_ego_contact_time_s'),
        planning_sampling='Every second canonical tick; cross-checked against the archived fixed-frame 10 Hz positions',
        planning_crosscheck_max_position_error_m=crosscheck,
        road_status='not_strictly_validated',other_actor_scope='Recorded observations only; missing observations remain unknown',
        limitations=list(validation.get('limitations',[])))
    return FrozenAttackScene(source_run,times,source_times,frame_ids,attack,attack_heading,attack_speed,attack_z,
        ego,np.unwrap(ego_heading),ego_speed,ego_z,metadata)


def _write_csv(path,fieldnames,rows):
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)


def write_frozen_input(scene,run_dir):
    """Create a new append-only input package; never overwrite an existing run."""
    run_dir=Path(run_dir);folder=run_dir/'input';folder.mkdir(parents=True,exist_ok=False)
    for name,view in [('attack-actual-20hz.csv',scene),('attack-actual-10hz.csv',scene.planning_view())]:
        rows=[]
        for i in range(len(view.times_s)):
            rows.append(dict(sample_index=i,frame_id=f'{view.frame_ids[i]:g}',execution_time_s=f'{view.times_s[i]:.12g}',
                source_time_s=f'{view.source_times_s[i]:.12g}',x_m=f'{view.attack_xy[i,0]:.12g}',y_m=f'{view.attack_xy[i,1]:.12g}',
                z_m=f'{view.attack_z_m[i]:.12g}',heading_rad=f'{view.attack_heading_rad[i]:.12g}',speed_mps=f'{view.attack_speed_mps[i]:.12g}'))
        _write_csv(folder/name,list(rows[0]),rows)
    view=scene.planning_view();rows=[]
    for i in range(len(view.times_s)):
        rows.append(dict(sample_index=i,frame_id=f'{view.frame_ids[i]:g}',execution_time_s=f'{view.times_s[i]:.12g}',
            source_time_s=f'{view.source_times_s[i]:.12g}',x_m=f'{view.ego_xy[i,0]:.12g}',y_m=f'{view.ego_xy[i,1]:.12g}',
            z_m=f'{view.ego_z_m[i]:.12g}',heading_rad=f'{view.ego_heading_rad[i]:.12g}',speed_mps=f'{view.ego_speed_mps[i]:.12g}'))
    _write_csv(folder/'ego-baseline.csv',list(rows[0]),rows)
    (folder/'frozen-input.json').write_text(json.dumps(metadata_json(scene),indent=2,ensure_ascii=False)+'\n')
    return folder


def metadata_json(scene):
    return dict(scene.metadata,canonical_samples=len(scene.times_s),planning_samples=len(scene.planning_view().times_s),
        execution_window_s=[float(scene.times_s[0]),float(scene.times_s[-1])],
        frame_range=[float(scene.frame_ids[0]),float(scene.frame_ids[-1])])
