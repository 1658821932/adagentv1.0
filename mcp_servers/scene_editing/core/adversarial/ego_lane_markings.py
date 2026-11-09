"""Check explicitly verified no-cross markings in the scene-local frame."""
import json
from pathlib import Path

import acceptance_policy as policy
import numpy as np
from shapely.geometry import LineString

from run_ego_reference import EGO_FRONT, EGO_REAR, EGO_WIDTH, polygon


def load_no_cross_lines(path):
    path = Path(path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if data.get('coordinate_frame') != 'scene_local' or not data.get('lines'):
        raise ValueError('Lane markings require scene_local coordinates and a nonempty lines list')
    lines = []
    for item in data['lines']:
        points = np.asarray(item['points'], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('Each lane marking requires at least two finite 2D points')
        line = LineString(points)
        if line.length == 0:
            raise ValueError('Lane marking must have nonzero length')
        lines.append((str(item['id']), line))
    return lines


def check_no_cross_lines(times, xy, yaw, lines):
    """Sample interpolated vehicle footprints; this is not a continuous proof."""
    sample_times = np.linspace(times[0], times[-1], int(np.ceil((times[-1] - times[0]) / policy.MAX_CHECK_STEP_S)) + 1)
    positions = np.column_stack([np.interp(sample_times, times, xy[:, axis]) for axis in range(2)])
    angles = np.interp(sample_times, times, np.unwrap(yaw))
    shift = (EGO_FRONT - EGO_REAR) / 2
    violations = []
    for time, position, angle in zip(sample_times, positions, angles):
        center = position + shift * np.array([np.cos(angle), np.sin(angle)])
        footprint = polygon(*center, angle, EGO_FRONT + EGO_REAR, EGO_WIDTH)
        for line_id, line in lines:
            if footprint.intersects(line):
                violations.append(dict(time_s=round(float(time), 6), line_id=line_id))
    return dict(passed=not violations, violations=violations, sample_interval_max_s=policy.MAX_CHECK_STEP_S)
