"""Recorded other-vehicle context and same-time contact screening (explicit opt-in)."""
import csv
import json
from pathlib import Path

import numpy as np

from trajectory_search_feedback import rectangle_overlap


POLICY='reject_before_or_at_first_ego_contact'


def build_context(track_csv, ego_pose_path, times, target_track_id, analysis_frame=4):
    with Path(track_csv).open(encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    poses=json.loads(Path(ego_pose_path).read_text())
    origin=np.asarray(poses['0'],dtype=float);inverse=np.linalg.inv(origin[:2,:2])
    times=np.asarray(times,dtype=float);actors={};seen=set()
    for row in rows:
        frame,tid=int(row['frame_id']),int(row['track_id']);index=frame-analysis_frame
        if row['object_class']!='vehicle' or tid==target_track_id or not 0<=index<len(times):
            continue
        if (frame,tid) in seen:
            raise ValueError('Duplicate other-vehicle observation')
        seen.add((frame,tid));pose=np.asarray(poses[str(frame)],dtype=float)
        xy=inverse@(pose[:2,:2]@np.array([float(row['box_center_x']),float(row['box_center_y'])])+pose[:2,3]-origin[:2,3])
        heading=float(row['box_heading'])
        direction=inverse@pose[:2,:2]@np.array([np.cos(heading),np.sin(heading)])
        sample=dict(sample_index=index,frame_id=frame,time_s=float(times[index]),x=float(xy[0]),y=float(xy[1]),
            heading_rad=float(np.arctan2(direction[1],direction[0])),length_m=float(row['length']),width_m=float(row['width']))
        if not np.isfinite(list(sample.values())).all() or sample['length_m']<=0 or sample['width_m']<=0:
            raise ValueError('Invalid other-vehicle geometry')
        actors.setdefault(tid,[]).append(sample)
    return dict(version='recorded_other_vehicles.v1',target_track_id=int(target_track_id),
        times_s=times.tolist(),coordinate_frame='Fixed frame0 x forward, y left',
        heading_source='Recorded box heading rotated by same-frame ego pose into frame0',
        missing_data='Unknown outside observed samples; no interpolation or extrapolation',
        actors=[dict(track_id=tid,samples=sorted(samples,key=lambda s:s['sample_index'])) for tid,samples in sorted(actors.items())])


def load_context(config, root):
    policy=config.get('other_actor_collision_policy')
    if policy is None:
        return None
    if policy!=POLICY:
        raise ValueError('Unknown other-vehicle collision policy')
    path=(Path(root)/config['other_actor_context']).resolve()
    if not path.is_relative_to((Path(root)/'experiments').resolve()):
        raise ValueError('Other-vehicle context must be an experiment artifact')
    context=json.loads(path.read_text())
    if context.get('version')!='recorded_other_vehicles.v1' or context['target_track_id']!=config['target_track_id']:
        raise ValueError('Other-vehicle context identity differs from run')
    return context


def intervals(times, ids):
    ids=np.asarray(ids,dtype=int)
    if not len(ids):return []
    groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    return [dict(start_s=float(times[g[0]]),end_s=float(times[g[-1]])) for g in groups]


def screen_other_actors(times, xy, headings, length, width, context, first_ego_contact):
    times=np.asarray(times,dtype=float)
    if context.get('version')!='recorded_other_vehicles.v1':
        raise ValueError('Unsupported other-vehicle context')
    np.testing.assert_allclose(context['times_s'],times,rtol=0,atol=1e-9)
    cutoff=float(times[-1]) if first_ego_contact is None else float(first_ego_contact)
    per_actor=[];violations=[];observed=0;track_ids=set()
    for actor in context['actors']:
        tid=actor['track_id']
        if tid==context['target_track_id'] or tid in track_ids:
            raise ValueError('Duplicate actor or edited target included in other-vehicle context')
        track_ids.add(tid)
        samples=actor['samples']
        if not samples:raise ValueError('Other vehicle has no observations')
        idx=np.array([s['sample_index'] for s in samples],dtype=int)
        if len(set(idx))!=len(idx) or np.any(np.diff(idx)<=0) or idx[0]<0 or idx[-1]>=len(times):
            raise ValueError('Invalid other-vehicle sample indices')
        np.testing.assert_allclose([s['time_s'] for s in samples],times[idx],rtol=0,atol=1e-9)
        data=np.array([[s[k] for k in ['x','y','heading_rad','length_m','width_m']] for s in samples])
        if not np.isfinite(data).all() or np.any(data[:,3:]<=0):
            raise ValueError('Invalid other-vehicle geometry')
        other=data[:,:2];angle=data[:,2];delta=xy[idx]-other
        c,s=np.cos(angle),np.sin(angle)
        relative=np.column_stack([c*delta[:,0]+s*delta[:,1],-s*delta[:,0]+c*delta[:,1]])
        contact=rectangle_overlap(relative,headings[idx]-angle,length,width,data[:,3],data[:,4])
        blocked=contact & (times[idx]<=cutoff+1e-9)
        hits=idx[contact];bad=idx[blocked]
        nearest=int(np.linalg.norm(delta,axis=1).argmin())
        report=dict(track_id=tid,observed_samples=len(idx),missing_samples=len(times)-len(idx),
            first_observed_s=float(times[idx[0]]),last_observed_s=float(times[idx[-1]]),
            minimum_center_distance_m=float(np.linalg.norm(delta[nearest])),closest_time_s=float(times[idx[nearest]]),
            overlap_intervals=intervals(times,hits),forbidden_overlap_intervals=intervals(times,bad),
            post_ego_contact_overlap_intervals=intervals(times,idx[contact & (times[idx]>cutoff+1e-9)]))
        per_actor.append(report);observed+=len(idx)
        for interval in report['forbidden_overlap_intervals']:
            violations.append(dict(metric='other_vehicle_overlap',track_id=tid,time_s=interval['start_s'],
                end_time_s=interval['end_s'],requirement='No other-vehicle overlap before or at first ego contact'))
    return dict(status='failed' if violations else 'passed',policy=POLICY,
        checked_until_s=cutoff,cutoff_basis='first sampled ego overlap' if first_ego_contact is not None else 'full reference window; no ego overlap',
        actors_checked=len(per_actor),observed_pair_samples=observed,
        missing_pair_samples=len(times)*len(per_actor)-observed,
        coverage='partial' if any(a['missing_samples'] for a in per_actor) else 'complete_for_listed_tracks',
        violations=violations,per_actor=per_actor,
        scope='10Hz oriented-rectangle checks against recorded other vehicles on observed samples only; missing observations are unknown. '
              'Post-ego-contact reference continuation is diagnostic, not multi-body physics. Pedestrians, unobserved actors and road boundaries are not checked.')


def prompt_summary(context, baseline, limit=6):
    summaries=[]
    for actor in context['actors']:
        samples=actor['samples'];idx=np.array([s['sample_index'] for s in samples])
        positions=np.array([[s['x'],s['y']] for s in samples])
        distances=np.linalg.norm(baseline[idx]-positions,axis=1);nearest=int(distances.argmin())
        keep=sorted(set([0,len(samples)-1,nearest,*range(0,len(samples),20)]))
        summaries.append(dict(track_id=actor['track_id'],observed_samples=len(samples),
            nearest_recorded_target_center_distance_m=round(float(distances[nearest]),3),
            samples=[{k:round(float(samples[i][k]),3) for k in ['time_s','x','y','heading_rad','length_m','width_m']} for i in keep]))
    summaries.sort(key=lambda a:a['nearest_recorded_target_center_distance_m'])
    return dict(actors_checked_by_program=len(summaries),displayed_nearest_actors=summaries[:limit],
        selection='Nearest to recorded target for input brevity only; program checks ALL listed vehicles',
        trajectory_source='Recorded future vehicle trajectories, held fixed during editing; this is playback-based planning context, not online prediction',
        missing_data=context['missing_data'],collision_policy=POLICY)
