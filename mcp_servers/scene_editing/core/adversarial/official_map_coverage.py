"""Expand legacy official map exports in an attack run's working scene."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from backend_configuration import load_settings
from experiment_runtime import atomic_json


ROAD_MAP = Path('metadata/map_drivable_official.json')
SOURCE_BINDING = 'official-map-source.json'
MIN_RADIUS_M = 300.0


def prepare_coverage(scene, output):
    """Keep input and checkpoint artifacts immutable, including on resume."""
    road_path = scene / ROAD_MAP
    reference_path = scene / 'metadata/map_reference.json'
    if not road_path.is_file() or not reference_path.is_file():
        return scene
    original_bytes = road_path.read_bytes()
    road = json.loads(original_bytes)
    if 'map_radius_m' not in road or not road.get('frame_convention', '').startswith('trajectory-local:'):
        return scene

    frames_path = scene / 'metadata/frames.json'
    frames = json.loads(frames_path.read_text())
    positions = np.array([np.asarray(frame['ego2global'])[:2, 3] for frame in frames], dtype=float)
    route_extent = float(np.linalg.norm(positions - positions[0], axis=1).max())
    radius = max(MIN_RADIUS_M, math.ceil((route_extent + 50.0) / 50.0) * 50.0)
    current_radius = float(road['map_radius_m'])
    if current_radius >= radius:
        return scene

    settings = load_settings()['pdm']
    reference = json.loads(reference_path.read_text())
    location = reference.get('map_location') or reference.get('map')
    if not location or road.get('map_location') != location:
        raise ValueError('Official map location differs from the scene map reference')
    map_root = Path(reference.get('maps_root') or settings['map_root'])
    if not (map_root / location).is_dir():
        map_root = Path(settings['map_root'])

    expanded = output / ROAD_MAP.name
    log = output / 'official-map-export.log'
    env = dict(os.environ, PYTHONPATH=settings['worldengine'], PYTHONDONTWRITEBYTECODE='1')
    env['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, [settings.get('system_lib'), env.get('LD_LIBRARY_PATH')]))
    with log.open('w') as stream:
        result = subprocess.run([settings['python'], str(Path(__file__).resolve()),
            str(road_path), str(map_root), str(radius), str(expanded)], env=env,
            stdout=stream, stderr=subprocess.STDOUT, timeout=90)
    if result.returncode:
        raise RuntimeError(f'Official road map expansion failed. See {log}')

    if not road_path.parent.resolve().is_relative_to(output):
        original = scene
        scene = output / 'scene_with_official_map'
        (scene / 'metadata').mkdir(parents=True)
        for entry in original.iterdir():
            if entry.name != 'metadata':
                (scene / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        for entry in (original / 'metadata').iterdir():
            if entry.name != ROAD_MAP.name:
                (scene / 'metadata' / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        atomic_json(scene / SOURCE_BINDING, dict(source_scene=str(original)))
    expanded.replace(scene / ROAD_MAP)
    atomic_json(output / 'official-map-coverage.json', dict(
        source_map_path=str(road_path), map_path=str(scene / ROAD_MAP),
        source_map_sha256=hashlib.sha256(original_bytes).hexdigest(),
        source_radius_m=current_radius, map_radius_m=radius, log_path=str(log)))
    return scene


def export_map(source, map_root, radius, destination):
    """Run under the existing WorldEngine environment using the same DAC map API."""
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap

    road = json.loads(source.read_text())
    origin = np.array(road['first_frame_rear_axle_global'], dtype=float)
    yaw = float(road['first_frame_yaw_rad'])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    ego = EgoState.build_from_rear_axle(rear_axle_pose=StateSE2(*origin, yaw),
        rear_axle_velocity_2d=StateVector2D(0., 0.), rear_axle_acceleration_2d=StateVector2D(0., 0.),
        tire_steering_angle=0., time_point=TimePoint(0), vehicle_parameters=get_pacifica_parameters())
    map_api = get_maps_api(map_root, 'nuplan-maps-v1.0', road['map_location'])
    drivable = PDMDrivableMap.from_simulation(map_api, ego, radius)

    def local(ring):
        return ((np.asarray(ring.coords)[:, :2] - origin) @ rotation).tolist()

    features = {}
    counts = {}
    for token, layer, geometry in zip(drivable._tokens, drivable.map_types, drivable._geometries):
        parts = geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry]
        for polygon in parts:
            if polygon.is_empty:
                continue
            key = f'{layer.name}:{token}:{len(features)}'
            features[key] = dict(type=layer.name, token=str(token), polygon=local(polygon.exterior))
            if polygon.interiors:
                features[key]['holes'] = [local(hole) for hole in polygon.interiors]
            counts[layer.name] = counts.get(layer.name, 0) + 1
    if not features:
        raise ValueError('Official map expansion returned no drivable polygons')
    road.update(features=features, map_radius_m=radius, polygon_type_counts=counts)
    destination.write_text(json.dumps(road))


if __name__ == '__main__':
    source, root, radius, destination = sys.argv[1:]
    export_map(Path(source), root, float(radius), Path(destination))
