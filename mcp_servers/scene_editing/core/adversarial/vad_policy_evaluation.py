"""Evaluate sampled VADv2 road compliance alongside native target collision."""

import hashlib
import math

from shapely.geometry import Point

from ego_road_checks import load_official_drivable
from policy_evidence import (
    EVALUATION_WINDOW, POLICY_EVALUATION_VERSION, confirmed_road_failure,
    in_policy_window, sampled_target_collision,
)
from box_geometry import corners


def road_compliance(feedback, road_map):
    if feedback.get('status') != 'measured' or not road_map.is_file():
        return dict(status='unavailable', reason='Missing native ego states or official map.')

    rows = feedback['full_precision_states']
    eligible = [row for row in rows if in_policy_window(row['time_s'])]
    start, end = EVALUATION_WINDOW
    if not eligible or rows[0]['time_s'] != 0 or eligible[-1]['time_s'] != end:
        return dict(status='unavailable', reason='Native ego states do not cover the evaluation window.')

    road = load_official_drivable(road_map)
    origin = rows[0]['ego_body_center_xy_m']
    try:
        radius = float(road.meta.get('map_radius_m'))
    except (TypeError, ValueError):
        radius = 0.
    if not math.isfinite(radius) or radius <= 0:
        radius = 0.
    violations = []
    unknown_frames = 0
    for row in eligible:
        box = [*row['ego_body_center_xy_m'], row['ego_length_m'], row['ego_width_m'],
               row['ego_heading_rad']]
        outside = []
        coverage_unknown = False
        for xy in corners(box):
            point = Point(xy)
            if road.union.covers(point):
                continue
            if math.dist(xy, origin) >= radius:
                coverage_unknown = True
                continue
            outside.append(float(road.union.distance(point)))
        unknown_frames += int(coverage_unknown)
        if outside:
            violations.append(dict(time_s=row['time_s'], step=row['step'],
                outside_corner_count=len(outside), worst_corner_outside_m=max(outside)))
    return dict(
        status='failed' if violations else 'inconclusive' if unknown_frames else 'passed',
        evaluated_frame_count=len(eligible), coverage_unknown_frame_count=unknown_frames,
        violation_frame_count=len(violations),
        first_violation_time_s=violations[0]['time_s'] if violations else None,
        worst_corner_outside_m=max((item['worst_corner_outside_m'] for item in violations), default=0.),
        violations=violations, map_query_radius_m=radius or None,
        map_sha256=hashlib.sha256(road_map.read_bytes()).hexdigest(),
        evaluation_window_s=[start, end],
        method='Actual ego body corners against official drivable polygons at recorded policy frames.',
    )


def evaluate_policy(collision, feedback, road_map):
    road = road_compliance(feedback, road_map)
    collision_failed = sampled_target_collision(collision)
    reasons = []
    if collision_failed:
        reasons.append('target_collision')
    if confirmed_road_failure(road):
        reasons.append('drivable_area_violation')
    return dict(schema_version=POLICY_EVALUATION_VERSION,
        status='failed' if reasons else 'passed' if road['status'] == 'passed' else 'inconclusive',
        policy_attack_success=bool(reasons), failure_reasons=reasons,
        target_collision_failed=collision_failed, drivable_area=road)


def compact_policy_evaluation(value):
    return dict(value, drivable_area={key: item for key, item in value['drivable_area'].items()
                                     if key != 'violations'})
