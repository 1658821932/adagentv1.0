"""Model-facing geometry and sampled collision evidence in the scene frame."""
import numpy as np
from shapely.ops import nearest_points
from run_ego_reference import EGO_FRONT, EGO_REAR, EGO_WIDTH, polygon
from ego_trajectory_edit import start_state


TOOL_SEMANTICS = dict(
    max_lateral_speed_mps='Absolute sampled displacement velocity projected onto the body left axis at the arithmetic midpoint of consecutive stored headings. This diagnoses body sideslip, not road-coordinate lateral travel speed; turning changes the body axes. Native interval sampling does not certify tire dynamics.',
    offset_bounds='The +/-2m forward/left and +/-0.35rad yaw bounds apply only to set_trajectory_offsets fixed-anchor parameters. Absolute position, path progress and braking/steering tools use their own contracts plus unchanged motion, road and contact checks.',
    search_evidence='Separate executed candidate checks, analytic approximations and untested maneuver ideas. Search failure is not a global infeasibility proof.')


def model_policy(effective):
    return {key:value for key,value in effective.items() if key != 'min_progress_fraction' and not key.startswith(('attack_','generator_'))}


def geometry_evidence(times, xy, yaw, actors, target_track, time_s):
    index = int(np.argmin(abs(np.asarray(times)-time_s)))
    position, angle = xy[index], float(yaw[index])
    direction = np.array([np.cos(angle), np.sin(angle)])
    center = position+(EGO_FRONT-EGO_REAR)/2*direction
    ego = polygon(*center, angle, EGO_FRONT+EGO_REAR, EGO_WIDTH)
    target = next((a for a in actors[index] if a['track']==target_track), None)
    result = dict(frame_index=index, observed_time_s=float(times[index]), requested_time_s=float(time_s),
        coordinate_frame='scene-local x/y, metres; ego pose is rear axle, actor poses are box centres',
        ego=dict(reference_point='rear_axle', xy_m=position.tolist(), yaw_rad=angle,
            dimensions_m=dict(front_from_reference=EGO_FRONT,rear_from_reference=EGO_REAR,width=EGO_WIDTH),
            center_xy_m=center.tolist(), front_center_xy_m=(position+EGO_FRONT*direction).tolist(),
            rear_center_xy_m=(position-EGO_REAR*direction).tolist(),corners_xy_m=list(map(list,ego.exterior.coords))[:-1]),
        target=None, scope='Nearest observed frame; exact start state is provided separately. No lane-line certification.')
    if target is not None:
        body = polygon(target['x'],target['y'],target['yaw'],target['length'],target['width'])
        first,second = nearest_points(ego,body)
        result.update(target=dict(track=target_track,reference_point='box_center',xy_m=[target['x'],target['y']],
            yaw_rad=target['yaw'],length_m=target['length'],width_m=target['width'],
            corners_xy_m=list(map(list,body.exterior.coords))[:-1],source='frozen_attack_target'),
            nearest_points_xy_m=[list(first.coords)[0],list(second.coords)[0]],
            clearance_m=float(ego.distance(body)),intersection_area_m2=float(ego.intersection(body).area))
    return result


def overlap_summary(times, areas):
    areas = np.asarray(areas)
    touching = np.flatnonzero(areas>1e-9)
    return dict(max_intersection_area_m2=float(areas.max()),
        sampled_area_integral_m2s=float(np.trapz(areas,times)),
        positive_area_frames=touching.tolist(),
        first_positive_area_sample_s=None if not len(touching) else float(times[touching[0]]),
        scope='Observed-frame geometric overlap; not collision damage or continuous-time contact duration')


def braking_diagnostic(times, xy, yaw, actors, target_track, effective):
    time = effective['observation_end_s']
    state = start_state(times,xy,yaw,time)
    geometry = geometry_evidence(times,xy,yaw,actors,target_track,time)
    speed = max(0.,state['signed_speed_mps']);accel = effective['max_acceleration_mps2']
    distance = speed**2/(2*accel)
    gap = geometry.get('clearance_m')
    from vehicle_trajectory import actor_observation_series, actor_pose_at_time
    dense = np.linspace(times[0],times[-1],int(np.ceil((times[-1]-times[0])/effective['max_check_step_s']))+1)
    first_contact = None
    start_gap = None
    for track,observed,poses in actor_observation_series(times,actors,'__none__'):
        if track != target_track: continue
        for query in np.unique(np.r_[dense,time]):
            pose = actor_pose_at_time(observed,poses,query,1.1*float(np.median(np.diff(times))),scene_times=times)
            if pose is None: continue
            angle = float(np.interp(query,times,np.unwrap(yaw)))
            position = np.array([np.interp(query,times,xy[:,k]) for k in range(2)])
            center = position+(EGO_FRONT-EGO_REAR)/2*np.array([np.cos(angle),np.sin(angle)])
            ego = polygon(*center,angle,EGO_FRONT+EGO_REAR,EGO_WIDTH)
            target = polygon(*pose[:2],pose[2],*pose[3:5])
            if abs(query-time)<1e-8: start_gap=float(ego.distance(target))
            if first_contact is None and ego.intersects(target): first_contact=float(query)
    return dict(start_state=state, geometry=geometry,
        edit_start_clearance_m=start_gap,
        baseline_first_contact=dict(time_s=first_contact,max_step_s=effective['max_check_step_s'],
            method='Uniform time grid plus edit boundary; piecewise linear ego and adjacent observed actor interpolation, polygon intersects. Sampled estimate, not exact first impact.'),
        defense_search=dict(status='not_searched',verified_passing_candidate_ids=[],global_feasibility_proven=False),
        diagnostic_only=True,attack_acceptance_affected=False,
        straight_line_stop_estimate=dict(speed_mps=speed,deceleration_mps2=accel,distance_m=distance,
            stop_time_s=time+speed/accel,nearest_observed_gap_m=gap,edit_start_clearance_m=start_gap,
            distance_exceeds_gap=None if start_gap is None else distance>start_gap,
            scope='Constant-deceleration longitudinal estimate, not proof about curved/lateral avoidance. '
                  'Do not use the first overlapping native sample as a continuous-time deadline.'))


def write_defense_difficulty(scene_dir, target_track, acceptance=None):
    """Write advisory evidence only after scene materialization; never gate an attack."""
    import json
    from pathlib import Path
    import acceptance_policy
    from experiment_runtime import atomic_json
    from run_ego_reference import scene_from_frames
    scene_dir=Path(scene_dir)
    frames=json.loads((scene_dir/'metadata/frames.json').read_text())
    times,xy,yaw,actors=scene_from_frames(frames,target_track)
    diagnostic=braking_diagnostic(times,xy,yaw,actors,target_track,acceptance_policy.resolve(acceptance))
    atomic_json(scene_dir/'defense-difficulty.json',diagnostic)
    return diagnostic


def defense_difficulty_or_unknown(scene_dir,target_track,acceptance=None):
    """Advisory failures must not change attack acceptance or materialization status."""
    try:
        return write_defense_difficulty(scene_dir,target_track,acceptance)
    except Exception as error:
        return dict(status='unavailable',error_type=type(error).__name__,diagnostic_only=True,
            attack_acceptance_affected=False,global_feasibility_proven=False)
