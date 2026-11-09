"""Authoritative PDM cross-check: score ego-editor trajectories with the full WorldEngine scorer.

Runs the *actual* WorldEngine PDMScorer (ghc complete copy, symlinked at
external/worldengine_ghc) on our recorded/edited trajectories, instead of our adapted
re-implementation (scripts/ego_pdms_score.py). Everything is built from the same scene
data (metadata/frames.json) transformed into the global nuPlan frame:

- expert proposal = recorded ego rear-axle trajectory (frames.json ego2global);
- pred proposal(s)  = candidate trajectory.csv (trajectory-local frame -> global via the
  first-frame ego pose), velocities from finite differences rotated into the ego frame;
- observations      = frames.json anns (ego/lidar frame boxes -> global centers),
  linear time grid of the 24 recorded frames (2 Hz), velocities finite-differenced;
- drivable map      = nuPlan map_api around the initial ego state (radius 100 m);
- route lanes       = roadblocks containing any recorded ego rear-axle position
  (approximation of their checkpoint_lanes route; only affects oncoming/DDC branches,
  which carry weight 0).

Documented deviations from the WorldEngine closed-loop metric pipeline:
- offline scoring of reference trajectories, not a simulated rollout;
- route roadblock ids approximated by proximity (see above);
- rear-axle accelerations are zeroed exactly like their MetricManager (comfort
  lon/lat/jerk sub-checks are therefore vacuous here, yaw metrics are real);
- agents are the 2 Hz recorded annotations only (no reactive replay).

Verified convention: frames.json ego2global translation is the ego REAR AXLE
(scenario pkl object_track positions are centers; per-frame difference equals the
Pacifica rear_axle_to_center distance 1.461 m along the heading to 1e-10).

Usage (requires the ghc simengine python with nuplan):
  LD_LIBRARY_PATH=$PWD/external/system-libs/usr/lib/x86_64-linux-gnu \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/external/worldengine_ghc/projects/SimEngine \
  /inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc/conda_envs/simengine/bin/python \
  scripts/run_we_pdm_crosscheck.py --run <run_dir> --output <new_dir>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

GHC_ROOT = Path('/inspire/hdd3/project/intelligent-driving-agent/public/workspace/ghc')
MAP_LOCATION_DEFAULT = 'us-pa-pittsburgh-hazelwood'

LABEL_TO_TYPE = {
    'vehicle': 'VEHICLE', 'pedestrian': 'PEDESTRIAN', 'bicycle': 'BICYCLE',
    'traffic_cone': 'TRAFFIC_CONE', 'barrier': 'BARRIER', 'czone_sign': 'CZONE_SIGN',
    'generic_object': 'GENERIC_OBJECT',
}
AGENT_TYPES = {'VEHICLE', 'PEDESTRIAN', 'BICYCLE'}


def load_scene_global(scene_dir):
    """Recorded ego rear-axle poses and per-frame annotations in the global nuPlan frame."""
    frames = json.loads((Path(scene_dir) / 'metadata/frames.json').read_text())
    ego_w = np.array([f['ego2global'] for f in frames], dtype=float)
    ego_xy = ego_w[:, :2, 3]
    ego_yaw = np.unwrap(np.arctan2(ego_w[:, 1, 0], ego_w[:, 0, 0]))
    timestamps = [int(f['timestamp']) for f in frames]
    detections = []
    for i, frame in enumerate(frames):
        anns = frame['anns']
        c, s = np.cos(ego_yaw[i]), np.sin(ego_yaw[i])
        rotation = np.array([[c, -s], [s, c]])
        frame_objects = []
        for token, label, box in zip(anns['track_tokens'], anns['gt_names'], anns['gt_boxes']):
            box = np.asarray(box, dtype=float)
            if box.shape[0] < 7 or not np.isfinite(box[:7]).all():
                continue
            center = ego_xy[i] + rotation @ box[:2]
            frame_objects.append({'track': str(token), 'label': str(label),
                                  'x': float(center[0]), 'y': float(center[1]),
                                  'yaw': float(ego_yaw[i] + box[6]),
                                  'length': float(box[3]), 'width': float(box[4]),
                                  'height': float(box[5]) if box.shape[0] > 5 else 2.0})
        detections.append(frame_objects)
    return timestamps, ego_xy, ego_yaw, detections


def finite_diff(times_s, values, edge_order=2):
    return np.gradient(values, times_s, axis=0, edge_order=edge_order)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', type=Path, default=None)
    parser.add_argument('--candidates', nargs='+', default=['constructor_012', 'edit_005'])
    parser.add_argument('--map-root', type=Path,
                        default=GHC_ROOT / 'WorldEngine/data/raw/nuplan/dataset/maps')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {args.output}')

    scene_dir = args.scene
    if scene_dir is None:
        config = json.loads((args.run / 'config.json').read_text())
        scene_dir = Path(config['input_dir'])
    map_reference = json.loads((Path(scene_dir) / 'metadata/map_reference.json').read_text())
    map_location = map_reference.get('map_location') or map_reference.get('map') or MAP_LOCATION_DEFAULT

    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.oriented_box import OrientedBox
    from nuplan.common.actor_state.scene_object import SceneObjectMetadata
    from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.tracked_objects import Agent, StaticObject, TrackedObjects
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from worldengine.components.agents.policy.pdm_planner.observation.pdm_observation import PDMObservation
    from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
    from worldengine.components.agents.policy.pdm_planner.scoring.pdm_scorer import PDMScorer
    from worldengine.components.agents.policy.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    from worldengine.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath

    timestamps, ego_xy, ego_yaw, detections = load_scene_global(scene_dir)
    times_s = (np.asarray(timestamps) - timestamps[0]) / 1e6
    n_frames = len(times_s)
    assert n_frames == 24 and np.allclose(np.diff(times_s), 0.5, atol=1e-3), 'expected 24 frames at 2 Hz'

    # ---- per-track velocities (global frame, finite differences; observed range only)
    track_series = {}
    for i, frame_objects in enumerate(detections):
        for obj in frame_objects:
            entry = track_series.setdefault(obj['track'], {'label': obj['label'], 'idx': [], 'xy': []})
            entry['idx'].append(i)
            entry['xy'].append((obj['x'], obj['y']))
    track_velocity = {}
    for token, entry in track_series.items():
        t = times_s[entry['idx']]
        xy = np.asarray(entry['xy'])
        vel = np.gradient(xy, t, axis=0, edge_order=1) if len(t) >= 2 else np.zeros_like(xy)
        track_velocity[token] = dict(zip(entry['idx'], vel))

    # ---- nuPlan DetectionsTracks per frame
    detection_tracks = []
    for i, frame_objects in enumerate(detections):
        objects = []
        for obj in frame_objects:
            otype = TrackedObjectType[LABEL_TO_TYPE[obj['label']]]
            box = OrientedBox(StateSE2(obj['x'], obj['y'], obj['yaw']), obj['length'], obj['width'], obj['height'])
            metadata = SceneObjectMetadata(timestamp_us=timestamps[i], token=obj['track'], track_id=None,
                                           track_token=obj['track'], category_name=obj['label'])
            if LABEL_TO_TYPE[obj['label']] in AGENT_TYPES:
                vx, vy = track_velocity[obj['track']].get(i, np.zeros(2))
                objects.append(Agent(otype, box, StateVector2D(float(vx), float(vy)), metadata))
            else:
                objects.append(StaticObject(otype, box, metadata))
        detection_tracks.append(DetectionsTracks(TrackedObjects(objects)))

    # ---- ego states and map
    vehicle_parameters = get_pacifica_parameters()
    v_global = finite_diff(times_s, ego_xy)
    v_lon0 = float(np.cos(ego_yaw[0]) * v_global[0, 0] + np.sin(ego_yaw[0]) * v_global[0, 1])
    v_lat0 = float(-np.sin(ego_yaw[0]) * v_global[0, 0] + np.cos(ego_yaw[0]) * v_global[0, 1])
    initial_ego_state = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(ego_xy[0, 0], ego_xy[0, 1], ego_yaw[0]),
        rear_axle_velocity_2d=StateVector2D(v_lon0, v_lat0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0, time_point=TimePoint(timestamps[0]),
        vehicle_parameters=vehicle_parameters)

    map_api = get_maps_api(str(args.map_root), 'nuplan-maps-v1.0', map_location)
    drivable_map = PDMDrivableMap.from_simulation(map_api, initial_ego_state, 100)

    # ---- route lanes: roadblocks touched by the recorded ego path
    roadblock_ids = []
    for x, y in ego_xy:
        near = map_api.get_proximal_map_objects(Point2D(float(x), float(y)), 1.0,
                                                [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR])
        for layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
            roadblock_ids.extend(obj.id for obj in near[layer])
    roadblock_ids = list(dict.fromkeys(roadblock_ids))
    route_lane_dict = {}
    for rb_id in roadblock_ids:
        block = map_api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK) or \
                map_api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if block is None:
            continue
        for lane in block.interior_edges:
            route_lane_dict[lane.id] = lane

    # ---- observation
    proposal_sampling = TrajectorySampling(num_poses=n_frames - 1, interval_length=0.5)
    observation = PDMObservation(proposal_sampling, proposal_sampling, 100, observation_sample_res=1)
    observation.update_detections_tracks(detection_tracks, [[] for _ in detection_tracks],
                                         route_lane_dict, compute_traffic_light_data=True)

    # ---- expert (recorded) states
    def states_from(times, xy, yaw):
        v_g = finite_diff(times, xy)
        c, s = np.cos(yaw), np.sin(yaw)
        states = np.zeros((len(times), 11), dtype=np.float64)
        states[:, 0], states[:, 1], states[:, 2] = xy[:, 0], xy[:, 1], np.unwrap(yaw)
        states[:, 3] = c * v_g[:, 0] + s * v_g[:, 1]
        states[:, 4] = -s * v_g[:, 0] + c * v_g[:, 1]
        return states

    expert_states = states_from(times_s, ego_xy, ego_yaw)
    centerline = PDMPath([StateSE2(float(x), float(y), float(h)) for x, y, h in zip(ego_xy[:, 0], ego_xy[:, 1], ego_yaw)])

    # ---- candidate trajectories (local frame -> global)
    yaw0 = ego_yaw[0]
    rot_back = np.array([[np.cos(yaw0), np.sin(yaw0)], [-np.sin(yaw0), np.cos(yaw0)]])

    def candidate_states(csv_path):
        with Path(csv_path).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        t = np.array([float(r['time_s']) for r in rows])
        local = np.array([[float(r['ego_x_m']), float(r['ego_y_m'])] for r in rows])
        yaw_local = np.array([float(r['ego_yaw_rad']) for r in rows])
        if len(t) != n_frames or np.abs(t - times_s).max() > 1e-6:
            raise ValueError(f'{csv_path} time grid mismatch')
        xy = local @ rot_back + ego_xy[0]
        yaw = yaw_local + yaw0
        return states_from(t, xy, yaw)

    names = ['recorded'] + list(args.candidates)
    all_states = [expert_states]
    for name in args.candidates:
        all_states.append(candidate_states(args.run / 'candidates' / name / 'trajectory.csv'))

    results = {}
    for idx, name in enumerate(names):
        states = np.stack([expert_states, all_states[idx]], axis=0)
        scorer = PDMScorer(proposal_sampling)
        scores = scorer.score_proposals(states, initial_ego_state, observation, centerline,
                                        route_lane_dict, drivable_map, map_api)
        pred = 1
        results[name] = {
            'score': float(scores[pred]),
            'no_at_fault_collisions': float(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, pred]),
            'drivable_area_compliance': float(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, pred]),
            'ego_progress': float(scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, pred]),
            'time_to_collision_within_bound': float(scorer._weighted_metrics[WeightedMetricIndex.TTC, pred]),
            'comfort': float(scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, pred]),
            'driving_direction_compliance': float(scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, pred]),
            'lane_keeping': float(scorer._weighted_metrics[WeightedMetricIndex.LANE_KEEPING, pred]),
            'time_to_at_fault_collision_s': float(scorer.time_to_at_fault_collision(pred)),
            'time_to_ttc_infraction_s': float(scorer.time_to_ttc_infraction(pred)),
            'expert_score_for_reference': float(scores[0]),
        }

    args.output.mkdir(parents=True)
    summary = {
        'status': 'completed',
        'scorer': 'WorldEngine PDMScorer (ghc complete copy via external/worldengine_ghc symlink)',
        'scorer_source': str(GHC_ROOT / 'WorldEngine'),
        'run': str(args.run), 'scene': str(scene_dir), 'map_location': map_location,
        'n_frames': n_frames, 'interval_s': 0.5, 'map_radius_m': 100,
        'n_route_roadblocks': len(roadblock_ids), 'n_route_lanes': len(route_lane_dict),
        'convention_check': 'frames ego2global = rear axle (verified vs pkl center, 1.461 m to 1e-10)',
        'deviations': [line.strip() for line in __doc__.split('deviations from the WorldEngine')[1]
                       .split('Verified convention')[0].splitlines()
                       if line.strip().startswith('-')],
        'results': results,
        'physical_execution': 'not_run',
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
