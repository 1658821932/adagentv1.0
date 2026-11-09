"""Screen observed-time ego retiming candidates against a single primary interaction target (default: the recorded target)."""
from workflow_timing import timed_operation

import argparse
import csv
import json
from itertools import zip_longest
from pathlib import Path

import acceptance_policy as policy
import numpy as np
from shapely.geometry import Polygon

from ego_road_checks import (RoadGeometry, check_drivable_area, check_drivable_area_official,
                             load_official_drivable, load_scene_road)

DEFAULT_TARGET_TRACK = None  # A target must be selected for each scene.
EGO_FRONT, EGO_REAR, EGO_WIDTH = 4.049, 1.127, 2.297


def smoothstep(u):
    u = np.clip(u, 0., 1.)
    return u**3 * (10. + u * (-15. + 6. * u))


def stop_source_times(times, start_s, duration_s):
    """Follow the recorded path while smoothly reducing its time rate to zero."""
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0) or duration_s <= 0 or start_s < times[0]:
        raise ValueError('Stop trajectory requires increasing times and a valid start/duration')
    rate = 1. - smoothstep((times-start_s)/duration_s)
    return times[0] + np.r_[0., np.cumsum(np.diff(times)*(rate[:-1]+rate[1:])/2)]


def progress_with_safe_stop(times, xy, recorded_progress, allow_safe_stop=False):
    """Apply the progress gate or an explicitly enabled terminal-stop exception."""
    times = np.asarray(times, dtype=float); xy = np.asarray(xy, dtype=float)
    motion = np.linalg.norm(np.diff(xy, axis=0), axis=1)/np.diff(times)
    traveled = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if traveled >= policy.MIN_PROGRESS_FRACTION*recorded_progress:
        return True, 'minimum_progress_met'
    if allow_safe_stop and len(motion) >= 2 and np.all(motion[-2:] <= policy.STOP_SPEED_MPS):
        return True, 'terminal_stop_exception'
    return False, 'minimum_progress_not_met'


def polygon(x, y, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    vertices = np.array([[-length/2, -width/2], [length/2, -width/2],
                         [length/2, width/2], [-length/2, width/2]])
    return Polygon(vertices @ np.array([[c, s], [-s, c]]) + [x, y])


def ego_path_from_frames(frames):
    """Read the observed ego path without requiring any other tracked actor."""
    timestamps = np.array([f['timestamp'] for f in frames], dtype=float)
    times = (timestamps - timestamps[0]) / 1e6
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError('Missing or non-increasing original timestamps')
    ego_world = np.array([f['ego2global'] for f in frames], dtype=float)
    global_yaw = np.unwrap(np.arctan2(ego_world[:, 1, 0], ego_world[:, 0, 0]))
    yaw0 = global_yaw[0]
    origin_rotation = np.array([[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]])
    ego_xy = (ego_world[:, :2, 3] - ego_world[0, :2, 3]) @ origin_rotation
    ego_yaw = global_yaw - yaw0
    return times, ego_xy, ego_yaw


def observed_objects_from_frames(frames):
    """Decode observed object boxes without requiring a preselected target."""
    times, ego_xy, ego_yaw = ego_path_from_frames(frames)
    objects = []
    for frame_index, frame in enumerate(frames):
        lidar_to_ego = np.array(frame['lidar2ego'], dtype=float)
        base_yaw = ego_yaw[frame_index] + np.arctan2(lidar_to_ego[1, 0], lidar_to_ego[0, 0])
        cs, sn = np.cos(ego_yaw[frame_index]), np.sin(ego_yaw[frame_index])
        ego_rotation = np.array([[cs, -sn], [sn, cs]])
        a = frame['anns']; frame_objects = []
        for token, label, box in zip(a['track_tokens'], a['gt_names'], a['gt_boxes']):
            box = np.asarray(box, dtype=float)
            if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                raise ValueError('Invalid object box')
            ego_xyz = lidar_to_ego @ np.r_[box[:3], 1.]
            xy = ego_xy[frame_index] + ego_rotation @ ego_xyz[:2]
            frame_objects.append(dict(track=str(token), label=str(label), x=float(xy[0]), y=float(xy[1]),
                yaw=float(base_yaw + box[6]), length=float(box[3]), width=float(box[4])))
        objects.append(frame_objects)
    return objects


def scene_from_frames(frames, target_track=DEFAULT_TARGET_TRACK):
    if not isinstance(target_track, str) or not target_track.strip():
        raise ValueError('target_track must identify a vehicle in the current scene')
    times, ego_xy, ego_yaw = ego_path_from_frames(frames)
    objects = observed_objects_from_frames(frames)
    if any(sum(actor['track'] == target_track for actor in frame) != 1 for frame in objects):
        raise ValueError('Target track must have exactly one observed box in each frame')
    return times, ego_xy, ego_yaw, objects


def evaluate(times, baseline_xy, baseline_yaw, objects, source_times, target_track=DEFAULT_TARGET_TRACK):
    if not isinstance(target_track, str) or not target_track.strip():
        raise ValueError('target_track must identify a vehicle in the current scene')
    xy = np.column_stack([np.interp(source_times, times, baseline_xy[:, k]) for k in range(2)])
    yaw = np.interp(source_times, times, baseline_yaw)
    center_shift = (EGO_FRONT-EGO_REAR)/2
    target_clearance, other_min, overlaps, coverage = [], [], [], []
    for i, (position, angle, actors) in enumerate(zip(xy, yaw, objects)):
        center = position + center_shift*np.array([np.cos(angle), np.sin(angle)])
        ego = polygon(*center, angle, EGO_FRONT+EGO_REAR, EGO_WIDTH)
        target = next(o for o in actors if o['track'] == target_track)
        target_poly = polygon(target['x'], target['y'], target['yaw'], target['length'], target['width'])
        target_clearance.append(float(ego.distance(target_poly)))
        if ego.intersects(target_poly): overlaps.append(i)
        others = [o for o in actors if o['track'] != target_track]
        distances = [(ego.distance(polygon(o['x'], o['y'], o['yaw'], o['length'], o['width'])), o) for o in others]
        if distances:
            val, obj = min(distances, key=lambda x:x[0]);other_min.append(float(val));coverage.append(len(others))
        else:other_min.append(None);coverage.append(0)
    dt = np.diff(times); velocity = np.diff(xy, axis=0)/dt[:,None]; speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(velocity, axis=0)/((dt[:-1]+dt[1:])/2)[:,None]
    distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    recorded_progress = float(np.linalg.norm(np.diff(baseline_xy,axis=0),axis=1).sum())
    progress = float(distances.sum())
    other_distances = [x for x in other_min if x is not None]
    return dict(xy=xy,yaw=yaw,target_clearance_m=target_clearance,other_min_m=other_min,
        target_overlap_indices=overlaps, other_observed_counts=coverage,
        min_target_clearance_m=float(min(target_clearance)),min_other_clearance_m=min(other_distances) if other_distances else None,
        max_speed_mps=float(speed.max()),max_acceleration_mps2=float(np.linalg.norm(acceleration,axis=1).max()),
        progress_m=progress,recorded_progress_m=recorded_progress)


def conflict_decisions(times, first_contact_s, effective, allow_stop=True):
    """A bounded, interleaved seed pool; Agent requests can refine all parameters."""
    boundary = effective['observation_end_s']
    end = float(times[-1])
    if boundary >= end:
        raise ValueError('No editable constructor horizon')
    deadline = end if first_contact_s is None else min(end, first_contact_s)
    span = max(0., deadline-boundary)
    starts = sorted(set(float(v) for v in np.linspace(boundary, boundary+span*.8, 4)))
    comfort = min(effective['comfort_reference_mps2'], effective['max_acceleration_mps2'])
    # Baseline is a real, checked candidate and a reusable construction parent.
    decisions = [('baseline', boundary, None)]
    for index,start in enumerate(starts):
        decisions.append(('retime', start, .5+index*.5))
        strengths = [comfort*.5, comfort, effective['max_acceleration_mps2']]
        for strength in strengths:
            if strength <= 0:
                continue
            # A moving terminal state is explored as well as a stop. These are
            # seed search proposals, never acceptance thresholds or required actions.
            decisions.append(('brake_profile', start, dict(deceleration_mps2=strength,
                target_speed_fraction=.5, lateral_offset_m=0.)))
            if allow_stop:
                decisions.append(('brake_profile', start, dict(deceleration_mps2=strength,
                    target_speed_mps=0., lateral_offset_m=0.)))
        # Existing steering capability remains in the pool, checked against map/actors.
        for offset in (-.5,.5):
            decisions.append(('brake_profile', start, dict(deceleration_mps2=comfort,
                target_speed_fraction=.5, lateral_offset_m=offset)))
    # Round-robin starts: an explicit cap must not discard every later intervention.
    baseline, remainder = decisions[:1], decisions[1:]
    groups = [[item for item in remainder if item[1] == start] for start in starts]
    def proposal_order(item):
        if item[0]=='retime':
            return 0
        value=item[2]
        moving='target_speed_fraction' in value
        strength=value['deceleration_mps2']
        if value.get('lateral_offset_m',0.):
            return 4
        if strength==comfort:
            return 2 if moving else 1
        if strength<comfort:
            return 3 if moving else 6
        return 7 if moving else 5
    for group in groups:
        group.sort(key=proposal_order)
    return baseline+[item for row in zip_longest(*groups)
                     for item in row if item is not None]


@timed_operation('ego.constructor', None)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--target-track',required=True)
    parser.add_argument('--allow-stop', action=argparse.BooleanOptionalAction, default=True,
                        help='Add smooth stop-and-hold candidates (default: enabled)')
    parser.add_argument('--acceptance-policy', type=Path, help='Named profile and task settings JSON')
    parser.add_argument('--max-candidates',type=int,help='Optional explicit numerical candidate cap; default preserves the full pool')
    args=parser.parse_args()
    if args.max_candidates is not None and args.max_candidates<1:
        raise ValueError('max-candidates must be positive')
    effective = policy.resolve(None if args.acceptance_policy is None else json.loads(args.acceptance_policy.read_text()))
    if args.acceptance_policy is None:
        effective.update(allow_safe_stop=args.allow_stop, minimum_progress_fraction=.5)
    folder=args.run/'reference_screen'
    if folder.exists():raise FileExistsError('Refusing to overwrite existing reference screen')
    folder.mkdir(parents=True)
    frames=json.loads((args.input/'metadata/frames.json').read_text())
    road=load_scene_road(args.input,frames)
    official = None
    if args.allow_stop:
        official_path = args.input/'metadata/map_drivable_official.json'
        if not official_path.is_file():
            raise ValueError('Safe-stop mode requires the official drivable map')
        official = load_official_drivable(official_path)
        road = RoadGeometry(official.union, map_source=str(official_path),
                            polygon_count=len(official.polygons))
    times,xy,yaw,objects=scene_from_frames(frames,args.target_track)
    baseline=evaluate(times,xy,yaw,objects,times,args.target_track)
    from ego_edit_evidence import braking_diagnostic
    diagnostic = braking_diagnostic(times,xy,yaw,objects,args.target_track,effective)
    first_contact = diagnostic['baseline_first_contact']['time_s']
    decisions = conflict_decisions(times,first_contact,effective,args.allow_stop)
    generated_decision_count=len(decisions)
    if args.max_candidates is not None: decisions=decisions[:args.max_candidates]
    from ego_lane_markings import load_no_cross_lines
    no_cross_lines = load_no_cross_lines(args.input/'metadata/no_cross_lines.json')
    from defense_behavior import DrivingObjective,adaptive_spec,preferences
    objective=DrivingObjective(times,xy,objects,args.target_track,effective)
    seed_count=max(1,int(len(decisions)*.75))
    ledger=[]; construction_failures=[]
    for idx,(mode,start,parameter) in enumerate(decisions,1):
        adaptive=False
        if idx>seed_count:
            proposal=adaptive_spec(ledger,decisions,effective['observation_end_s'],float(times[-1]))
            if proposal is not None:
                mode,start,parameter=proposal;decisions[idx-1]=proposal;adaptive=True
        curve = operation = None
        if mode == 'brake_profile':
            from ego_trajectory_edit import brake_along_path
            from ego_trajectory_edit import start_state
            parameter = dict(parameter)
            if 'target_speed_fraction' in parameter:
                speed=float(np.linalg.norm(start_state(times,xy,yaw,start)['velocity_xy_mps']))
                parameter['target_speed_mps']=speed*parameter.pop('target_speed_fraction')
            stopping=parameter['target_speed_mps']==0
            operation=dict(edit_mode=mode,parent_candidate_id='recorded_baseline',path_candidate_id='recorded_baseline',
                start_time_s=start,motion_intent='forward_stop' if stopping else 'forward',
                end_behavior='hold' if stopping else 'cruise', path_geometry_version='merge_micro_knots.v1',
                reason='Scene-timed constructor seed; Agent may refine its timing and speed',**parameter)
            try:
                candidate_xy,candidate_yaw,curve=brake_along_path(times,xy,yaw,xy,yaw,operation,
                    effective['observation_end_s'],with_details=True)
            except ValueError as error:
                construction_failures.append(dict(operation=operation,error=str(error),status='construction_rejected'))
                continue
            duration=curve['transition_time_s']-start;delay=None;source_times=times.copy()
        elif mode == 'baseline':
            duration=0.;delay=None;source_times=times.copy()
        elif mode == 'stop':
            duration = parameter; delay = None; source_times = stop_source_times(times, start, duration)
        else:
            delay=parameter; duration=max(3.,1.875*delay+0.5); source_times=times-delay*smoothstep((times-start)/duration)
        if np.any(np.diff(source_times)<-1e-9) or source_times.min() < -1e-9:raise ValueError('Non-monotonic retiming')
        result=evaluate(times,xy,yaw,objects,source_times,args.target_track) if curve is None else evaluate(times,candidate_xy,candidate_yaw,objects,times,args.target_track)
        result['recorded_progress_m']=baseline['recorded_progress_m']
        other=result['min_other_clearance_m'];target_ok=not result['target_overlap_indices']
        other_ok=not any(result['other_observed_counts']) or (other is not None and other>0.)
        motion_ok=result['max_speed_mps']<=15 and result['max_acceleration_mps2']<=6
        progress_ok,progress_reason=progress_with_safe_stop(times,result['xy'],result['recorded_progress_m'],args.allow_stop)
        road_result=check_drivable_area(times,result['xy'],result['yaw'],road);road_ok=road_result['passed']
        official_result=(check_drivable_area_official(times,result['xy'],result['yaw'],official,min_margin_m=.1)
                         if official is not None else None)
        if official_result is not None:
            road_ok = road_ok and official_result['passed']
        record={k:v for k,v in result.items() if k not in ('xy','yaw','target_clearance_m','other_min_m','other_observed_counts')}
        from ego_action_tool import assess
        shared = assess(times, xy, yaw, objects, result['xy'], result['yaw'], 'retime',
                        target_track=args.target_track, road=road, road_official=official, allow_safe_stop=args.allow_stop, no_cross_lines=no_cross_lines, acceptance=effective)
        if curve is not None:
            from ego_trajectory_edit import apply_intent,apply_curve_checks
            apply_curve_checks(shared,curve)
            apply_intent(shared,times,result['xy'],result['yaw'],operation)
        motion_ok=all(shared['checks'][key] for key in ('speed','acceleration','reverse_speed','lateral_motion'))
        record['shared_checks'] = shared['checks']
        record.update(submit_status=shared['submit_status'], coverage=shared['coverage'],
                      input_qualification=shared['input_qualification'], full_window_status=shared['full_window_status'])
        record.update(behavior_metrics=objective.evaluate(result['xy'],result['yaw'],result['target_clearance_m']),
            adaptive_proposal=adaptive,operation=operation,generation_curve=curve,
            candidate_purpose='avoidance_attempt' if first_contact is None or start<first_contact else 'overlap_mitigation_after_baseline_contact')
        record.update(candidate_id=f'candidate_{idx:03d}',mode=mode,start_s=start,max_delay_s=delay,duration_s=duration,
            target_screen_passed=target_ok,other_observed_screen_passed=other_ok,motion_screen_passed=motion_ok,
            progress_screen_passed=progress_ok,progress_reason=progress_reason,road_screen_passed=road_ok,
            road_official=official_result,
            reference_screen_passed=shared['status']=='reference_screen_passed',
            road_status='passed' if road_ok else 'failed',road=road_result,physical_execution='not_run')
        target=folder/record['candidate_id'];target.mkdir()
        if curve is not None:
            (target/'curve-check.json').write_text(json.dumps(curve,indent=2)+'\n')
        with (target/'trajectory.csv').open('w',newline='') as stream:
            w=csv.writer(stream);w.writerow(['frame_index','time_s','source_ego_time_s','ego_x_m','ego_y_m','ego_yaw_rad','target_clearance_m','nearest_other_clearance_m','other_observed_count'])
            for i in range(len(times)):w.writerow([i,times[i],source_times[i],*result['xy'][i],result['yaw'][i],result['target_clearance_m'][i],result['other_min_m'][i],result['other_observed_counts'][i]])
        (target/'screen.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n');ledger.append(record)
    passing=[row['candidate_id'] for row in ledger if row['reference_screen_passed']]
    diagnostic['defense_search'].update(status='found_passing_candidate' if passing else 'searched_no_passing_candidate',verified_passing_candidate_ids=passing)
    (folder/'defense-difficulty.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    summary={'candidate_budget':dict(max_candidates=args.max_candidates,available_decisions=generated_decision_count), 'version':'ego_constructor.v4','behavior_preferences':preferences(),'adaptive_proposal_count':sum(r.get('adaptive_proposal',False) for r in ledger),'construction_failures':construction_failures,'defense_difficulty':diagnostic,'status':'reference_screen_complete','carla':'disabled','image_render':'not_started','sample_rate':'observed 2 Hz; timestamps preserved','method':'Conflict-timed retiming plus dynamically consistent braking and steering along recorded reference geometry; interpolated ego poses are generated hypotheses, target and other actors use only observed frames',
        'acceptance_policy':effective, 'target_track':args.target_track, 'allow_safe_stop':effective['allow_safe_stop'],
        'road_source':'official_union' if official is not None else 'scene_lane_union',
        'ego_geometry':'WorldEngine Pacifica rear axle, length 5.176 m width 2.297 m; assumption, not source-calibrated',
        'baseline_min_target_clearance_m':baseline['min_target_clearance_m'],'baseline_target_overlap_indices':baseline['target_overlap_indices'],
        'baseline_min_other_clearance_m':baseline['min_other_clearance_m'],'thresholds':{'target_contact':'forbidden','observed_other_contact':'forbidden','speed_mps':effective['max_speed_mps'],'acceleration_mps2':effective['max_acceleration_mps2'],'minimum_progress_fraction':effective['minimum_progress_fraction']}, 'clearance_policy':'continuous optimization metric, no minimum-distance gate',
        'candidates':ledger,'qualified_count':sum(c['reference_screen_passed'] for c in ledger),
        'limits':'Sampled 2 Hz actor checks and whole-body road checks densified to 0.1 s; no continuous collision proof or physical execution; ego timing edited, other actors recorded only; camera images not updated; progress metrics are cumulative traveled path length, not net forward displacement'}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':summary['status'],'baseline_min_target_clearance_m':summary['baseline_min_target_clearance_m'],'qualified_count':summary['qualified_count'],'candidates':len(ledger)},ensure_ascii=False))


if __name__=='__main__':main()
