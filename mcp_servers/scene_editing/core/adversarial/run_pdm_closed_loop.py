"""Local PDM closed loop: recorded 3 s history, PDM at 2 Hz, nuPlan LQR/bicycle at 10 Hz.

Uses WorldEngine's real PDMClosedPlanner without the SimEngine engine. Its fork
returns an Nx11 state array, adapted through its own state_array_to_ego_state.
Environment snapshots replay at original timestamps; no future ego states enter
planning. All external sources are read-only; map DBs are copied into the run.
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from run_we_pdm_crosscheck import load_scene_global, LABEL_TO_TYPE, AGENT_TYPES


def is_ego_annotation(obj):
    """Identity excludes ego; geometric overlap is valid attack evidence."""
    names = {'ego', 'av', 'sdc'}
    return str(obj['track']).lower() in names or str(obj['label']).lower() in names


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Original scene as read-only data source; not simulated by default')
    parser.add_argument('--baseline-diagnostic', action='store_true', help='Optionally simulate original scene for diagnosis; never an attack acceptance gate')
    parser.add_argument('--attack', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--map-root', type=Path, required=True)
    parser.add_argument('--target-track', default='synthetic_vehicle_001')
    parser.add_argument('--target-is-new', action='store_true')
    parser.add_argument('--history-s', type=float, default=3.)
    parser.add_argument('--baseline-cache', type=Path, help='Prior PDM output; reuse only a matching valid baseline')
    parser.add_argument('--ignore-traffic-lights', action='store_true',
                        help='Omit signals from PDM without changing annotations; signal compliance not evaluated')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from nuplan.common.actor_state.scene_object import SceneObjectMetadata
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.tracked_objects import Agent, StaticObject, TrackedObjects
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
    from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
    from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
    from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
    from worldengine.components.agents.policy.pdm_planner.pdm_closed_planner import PDMClosedPlanner
    from worldengine.components.agents.policy.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
    from worldengine.components.agents.policy.pdm_planner.utils.pdm_array_representation import state_array_to_ego_state
    from shapely.ops import unary_union
    from shapely.geometry import shape

    from pdm_temporal import align_takeover, RecordedActorReplay
    from pdm_binding import binding, matching_baseline
    settings = json.loads((Path(__file__).resolve().parent/'backend_settings.json').read_text())['pdm']
    bindings = binding(dict(challenge_baseline_scene_path=str(args.baseline),
        challenge_attack_scene_path=str(args.attack),challenge_history_s=args.history_s,
        challenge_target_track=args.target_track,challenge_target_is_new=args.target_is_new,
        challenge_ignore_traffic_lights=args.ignore_traffic_lights,
        challenge_run_baseline_diagnostic=args.baseline_diagnostic),settings)
    dump(args.output/'binding.json',bindings)
    baseline_reused = bool(args.baseline_diagnostic and args.baseline_cache and matching_baseline(args.baseline_cache,bindings))
    vehicle = get_pacifica_parameters()
    base_frames = json.loads((args.baseline / 'metadata/frames.json').read_text())
    attack_frames = json.loads((args.attack / 'metadata/frames.json').read_text())
    from pdm_traffic_lights import traffic_light_policy
    light_policy = traffic_light_policy(dict(baseline=base_frames, attack=attack_frames),
                                        ignore=args.ignore_traffic_lights)
    takeover = align_takeover([frame['timestamp'] for frame in base_frames],args.history_s)
    assert [f['timestamp'] for f in base_frames] == [f['timestamp'] for f in attack_frames]
    assert all(np.array_equal(a['ego2global'], b['ego2global']) for a, b in zip(base_frames, attack_frames))
    location = base_frames[0]['map_location']
    # Separate copy prevents SQLite WAL/SHM files in the shared map source.
    local_maps = args.output / 'maps'
    local_maps.mkdir()
    map_metadata = json.loads((args.map_root / 'nuplan-maps-v1.0.json').read_text())
    dump(local_maps / 'nuplan-maps-v1.0.json', {location: map_metadata[location]})
    shutil.copytree(args.map_root / location, local_maps / location)
    # nuPlan eagerly touches all cities; restrict its process-local loader to this copied city.
    import nuplan.database.maps_db.gpkg_mapsdb as maps_module
    maps_module.MAP_LOCATIONS = (location,)
    map_api = get_maps_api(str(local_maps), 'nuplan-maps-v1.0', location)
    route = base_frames[0]['roadblock_ids']  # supplied navigation route, no future ego proximity inference
    pdm_config = dict(trajectory_poses=50, proposal_poses=40, interval_s=.5,
                      speed_limit_fraction=[.15, .25, .5, .75, 1.0], fallback_target_velocity=30.,
                      min_gap_to_lead_agent=10., headway_time=1.5, accel_max=4.5, decel_max=3.,
                      lateral_offsets=[-3.5, 3.5], map_radius_m=100.)
    tracker_config = dict(q_longitudinal=[10.], r_longitudinal=[1.], q_lateral=[1., 10., 0.],
                          r_lateral=[1.], discretization_time=.1, tracking_horizon=10,
                          jerk_penalty=1e-4, curvature_rate_penalty=1e-2,
                          stopping_proportional_gain=1., stopping_velocity=.2)
    dump(args.output / 'config.json', dict(source_scene=str(args.baseline), attack=str(args.attack),
         baseline_diagnostic=args.baseline_diagnostic, baseline_is_acceptance_gate=False,
         source_map_root=str(args.map_root), local_map_root=str(local_maps), route_roadblocks=route,
         target_track=args.target_track, target_is_new=args.target_is_new,
         planner=pdm_config, tracker=tracker_config, observation_end_s=args.history_s,
         takeover_alignment=takeover,
         background_geometry_policy='Recorded linear interpolation for diagnostics only; planner inputs stay native causal snapshots',
         control_substeps_per_observation=5, architecture='WE PDMClosedPlanner + nuPlan TwoStageController/LQR/KinematicBicycleModel; no SimEngine engine',
         observation_policy='2Hz current snapshots; causal backward-difference velocities; zero-order-held between recorded snapshots; no missing tracks invented',
         traffic_lights=light_policy, ego_exclusion='explicit ego token labels and rear-axle-relative footprint check'))
    results = {}
    runs = [('attack', args.attack, attack_frames)]
    if args.baseline_diagnostic:
        runs.insert(0, ('baseline', args.baseline, base_frames))
    for name, scene, frames in runs:
        run_start = time.perf_counter()
        out = args.output / name
        if name == 'baseline' and baseline_reused:
            shutil.copytree(args.baseline_cache/'baseline',out)
            results[name]=json.loads((out/'summary.json').read_text())
            print('baseline reused',flush=True)
            continue
        out.mkdir()
        timestamps, xy, yaw, detections = load_scene_global(scene)
        times = (np.asarray(timestamps) - timestamps[0]) / 1e6
        assert np.max(abs(times - np.arange(len(times)) * .5)) < .001
        assert len(frames) >= 8
        takeover_index = takeover['index']
        assert 1 <= takeover_index < len(frames)-1
        takeover_us = takeover['takeover_time_us']
        # Current/previous samples only: no np.gradient central differences.
        last = {}
        observed = []
        coincident_reference_centers = []
        for i, objects in enumerate(detections):
            current = []
            for obj in objects:
                token = obj['track']
                # Ego is not normally an annotation. Refuse any accidental inclusion.
                if is_ego_annotation(obj):
                    continue
                h = yaw[i]
                center_ego = xy[i] + vehicle.rear_axle_to_center * np.array([np.cos(h), np.sin(h)])
                if np.linalg.norm(np.array([obj['x'], obj['y']]) - center_ego) < .25:
                    coincident_reference_centers.append(dict(frame_index=i, track=token,
                        time_s=float(times[i]), note='Geometric proximity is not ego identity; object retained.'))
                v = np.zeros(2)
                if token in last and last[token][0] == i - 1:
                    v = (np.array([obj['x'], obj['y']]) - last[token][1]) / (times[i] - times[i-1])
                last[token] = (i, np.array([obj['x'], obj['y']]))
                obj = dict(obj, vx=float(v[0]), vy=float(v[1]))
                current.append(obj)
            observed.append(current)
        if name == 'attack':
            target = [next(o for o in obs if o['track'] == args.target_track) for obs in observed]
        elif args.target_is_new:
            assert not any(o['track'] == args.target_track for obs in observed for o in obs)

        dump(out/'reference_center_proximity.json', coincident_reference_centers)

        def observation(i):
            objects = []
            for obj in observed[i]:
                otype = TrackedObjectType[LABEL_TO_TYPE[obj['label']]]
                box = OrientedBox(StateSE2(obj['x'], obj['y'], obj['yaw']), obj['length'], obj['width'], obj['height'])
                meta = SceneObjectMetadata(timestamp_us=timestamps[i], token=obj['track'], track_id=None,
                                           track_token=obj['track'], category_name=obj['label'])
                objects.append(Agent(otype, box, StateVector2D(obj['vx'], obj['vy']), meta)
                               if LABEL_TO_TYPE[obj['label']] in AGENT_TYPES else StaticObject(otype, box, meta))
            return DetectionsTracks(TrackedObjects(objects))

        def recorded_state(i):
            dyn = frames[i]['ego_dynamic_state']
            # Openscene dynamic state [ego-frame vx,vy,ax,ay].
            return EgoState.build_from_rear_axle(StateSE2(*xy[i], yaw[i]), StateVector2D(*dyn[:2]),
                       StateVector2D(*dyn[2:4]), 0., TimePoint(timestamps[i]), vehicle)

        states = [recorded_state(i) for i in range(takeover_index+1)]
        obs_history = [observation(i) for i in range(takeover_index+1)]
        # Native alignment never starts before the requested boundary. If no close
        # future native sample exists, advance only the last measured state causally.
        takeover_offset = (takeover_us-timestamps[takeover_index])/1e6
        measured = states[-1]
        if takeover_offset > 0:
            h = measured.rear_axle.heading
            v = measured.dynamic_car_state.rear_axle_velocity_2d
            pose = StateSE2(measured.rear_axle.x+takeover_offset*(np.cos(h)*v.x-np.sin(h)*v.y),
                            measured.rear_axle.y+takeover_offset*(np.sin(h)*v.x+np.cos(h)*v.y), h)
            takeover_state = EgoState.build_from_rear_axle(pose, v, measured.dynamic_car_state.rear_axle_acceleration_2d,
                              0., TimePoint(takeover_us), vehicle)
            states.append(takeover_state)
            obs_history.append(observation(takeover_index))
        planner = PDMClosedPlanner(TrajectorySampling(num_poses=50, interval_length=.5),
                 TrajectorySampling(num_poses=40, interval_length=.5),
                 BatchIDMPolicy(30., [.15, .25, .5, .75, 1.], 10., 1.5, 4.5, 3.), [-3.5, 3.5], 100.)
        planner.initialize(dict(map_api=map_api, route_roadblock_dict_ids=route))
        controller = TwoStageController(SimpleNamespace(initial_ego_state=states[-1]),
                      LQRTracker(**tracker_config), KinematicBicycleModel(vehicle))
        rows, plan_rows, evidence = [], [], []
        origin, heading0 = xy[0], yaw[0]
        rotation = np.array([[np.cos(heading0), -np.sin(heading0)], [np.sin(heading0), np.cos(heading0)]])
        def row(state, source):
            pos = (state.rear_axle.point.array - origin) @ rotation
            d = state.dynamic_car_state
            return dict(time_s=(state.time_point.time_us-timestamps[0])/1e6,
                 ego_x_m=float(pos[0]), ego_y_m=float(pos[1]), ego_yaw_rad=float(state.rear_axle.heading-heading0),
                 global_x_m=state.rear_axle.x, global_y_m=state.rear_axle.y,
                 speed_mps=float(d.rear_axle_velocity_2d.magnitude()), longitudinal_velocity_mps=d.rear_axle_velocity_2d.x,
                 acceleration_mps2=float(d.rear_axle_acceleration_2d.magnitude()),
                 tire_steering_rad=state.tire_steering_angle, angular_velocity_radps=d.angular_velocity, source=source)
        rows.extend(row(s, 'recorded_history') for s in states)
        roll_states = list(states)
        plan_times, control_times = [], []
        for i in range(takeover_index, len(frames)-1):
            current = states[-1]
            history = SimulationHistoryBuffer.initialize_from_list(7, states[-7:], obs_history[-7:])
            inp = PlannerInput(SimulationIteration(current.time_point, i), history, [])
            before = time.perf_counter()
            array = planner.compute_planner_trajectory(inp)
            plan_times.append(time.perf_counter()-before)
            assert isinstance(array, np.ndarray) and array.ndim == 2 and array.shape[1] == 11
            assert np.isfinite(array).all()
            planned = [state_array_to_ego_state(a, TimePoint(current.time_point.time_us+int(j*500000)), vehicle)
                       for j, a in enumerate(array)]
            trajectory = InterpolatedTrajectory(planned)
            for j, state in enumerate(planned):
                plan_rows.append(dict(iteration=i, horizon_index=j, **row(state, 'pdm_plan')))
            obs = observed[i]
            pdm_tokens = list(planner._observation._unique_objects.keys())
            pdm_collided = list(planner._observation.collided_track_ids)
            if name == 'attack':
                assert args.target_track in {o['track'] for o in obs}
                assert args.target_track in pdm_tokens or args.target_track in pdm_collided
            evidence.append(dict(iteration=i, current_time_s=(current.time_point.time_us-timestamps[0])/1e6, max_history_time_us=history.current_state[0].time_point.time_us,
                observation_timestamp_us=timestamps[i], planned_state_count=len(planned), planner_iteration=planner._iteration,
                observation_objects=obs, pdm_object_tokens=pdm_tokens, pdm_collided_track_ids=pdm_collided,
                target_present=any(o['track']==args.target_track for o in obs), target_in_pdm=args.target_track in pdm_tokens,
                route_roadblocks=list(planner._route_roadblock_dict), planning_seconds=plan_times[-1]))
            start_us, end_us = current.time_point.time_us, timestamps[i+1]
            for substep in range(5):
                now = TimePoint(start_us+round((end_us-start_us)*substep/5))
                nxt = TimePoint(start_us+round((end_us-start_us)*(substep+1)/5))
                before = time.perf_counter()
                controller.update_state(SimulationIteration(now, substep), SimulationIteration(nxt, substep+1), current, trajectory)
                control_times.append(time.perf_counter()-before)
                current = controller.get_state()
                rows.append(row(current, 'nuplan_controller'))
                roll_states.append(current)
            states.append(current)
            obs_history.append(observation(i+1))
            dump(out / 'progress.json', dict(last_iteration=i, rows=len(rows)))
        write_csv(out/'trajectory.csv', rows)
        write_csv(out/'plans.csv', plan_rows)
        dump(out/'observations.json', evidence)
        # Independent geometric diagnostics use actual propagated ego footprints.
        road_json = json.loads((scene/'metadata/map_drivable_official.json').read_text())
        dump(out/'road-schema.json', dict(type=type(road_json).__name__, keys=list(road_json) if isinstance(road_json, dict) else []))
        diagnostics = []
        actor_replay = RecordedActorReplay(times,observed)
        from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
        road_map = PDMDrivableMap.from_simulation(map_api, states[0], 100.)
        road = unary_union(road_map._geometries)
        for state, record in zip(roll_states, rows):
            t = record['time_s']
            truth_objects, replay_coverage = actor_replay.sample(t)
            i = replay_coverage['observation_frame']
            ego_poly = state.car_footprint.geometry
            gaps, overlaps = [], []
            for obj in truth_objects:
                poly = OrientedBox(StateSE2(obj['x'], obj['y'], obj['yaw']), obj['length'], obj['width'], obj['height']).geometry
                gaps.append((ego_poly.distance(poly), obj['track']))
                area = ego_poly.intersection(poly).area
                if area > 1e-6:
                    overlaps.append(dict(track=obj['track'], area_m2=area))
            gap, track = min(gaps) if gaps else (None, None)
            target_obj = next((o for o in truth_objects if o['track']==args.target_track), None)
            target_gap = None
            if target_obj:
                o = target_obj
                target_gap = ego_poly.distance(OrientedBox(StateSE2(o['x'],o['y'],o['yaw']),o['length'],o['width'],o['height']).geometry)
            diagnostics.append(dict(time_s=t, min_gap_m=gap, closest_track=track, target_gap_m=target_gap,
                contacts=overlaps, outside_road_area_m2=ego_poly.difference(road).area,
                road_covers_footprint=bool(road.buffer(.05).covers(ego_poly)), observation_frame=i,
                background_replay_coverage=replay_coverage))
        dump(out/'diagnostics.json', diagnostics)
        controlled = [r for r in rows if r['source']=='nuplan_controller']
        contacts = [d for d in diagnostics if d['contacts']]
        summary = dict(status='completed', trajectory_rows=len(rows), planning_calls=len(plan_times), initialize_calls=1,
            takeover_s=(takeover_us-timestamps[0])/1e6, requested_takeover_s=args.history_s,
            takeover_alignment_delay_s=takeover['alignment_delay_s'],
            last_history_sample_s=float(times[takeover_index]), causal_takeover_advance_s=takeover_offset,
            control_steps=len(control_times), min_gap_m=min(d['min_gap_m'] for d in diagnostics if d['min_gap_m'] is not None),
            min_target_gap_m=min((d['target_gap_m'] for d in diagnostics if d['target_gap_m'] is not None), default=None),
            contact_samples=len(contacts), first_contact=contacts[0] if contacts else None,
            road_violation_samples=sum(not d['road_covers_footprint'] for d in diagnostics),
            max_outside_road_area_m2=max(d['outside_road_area_m2'] for d in diagnostics),
            max_speed_mps=max(r['speed_mps'] for r in controlled), max_acceleration_mps2=max(r['acceleration_mps2'] for r in controlled),
            min_longitudinal_velocity_mps=min(r['longitudinal_velocity_mps'] for r in controlled),
            final_speed_mps=rows[-1]['speed_mps'], max_abs_steering_rad=max(abs(r['tire_steering_rad']) for r in controlled),
            planning_seconds_total=sum(plan_times), planning_seconds_max=max(plan_times), control_seconds_total=sum(control_times),
            scenario_wall_seconds=time.perf_counter()-run_start,
            collision_claim='2D positive-area box overlap in kinematic rollout with recorded background replay; no rigid-body contact physics')
        dump(out/'summary.json', summary)
        results[name] = summary
        print(name, json.dumps(summary), flush=True)
    dump(args.output/'summary.json', dict(status='completed', wall_seconds=time.perf_counter()-started,
        traffic_light_policy=light_policy, takeover_alignment=takeover,
        background_geometry_policy='linear_recorded; evaluation only, native causal planner snapshots',
        baseline_reused=baseline_reused,
        baseline_cache_source=str(args.baseline_cache) if baseline_reused else None, results=results))


if __name__ == '__main__':
    main()
