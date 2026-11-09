"""Strict parsing for structured ego defense decisions."""
import math

DEFENSE_DECISION_VERSION = 'defense_decision.v1'
ACTION_TYPES = frozenset({'brake', 'lane_change', 'brake_then_lane_change'})
ALLOWED_STATUSES = ('success', 'candidate_failed', 'undetermined', 'unavoidable')


def _fields(value, name, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError(name + ' must be an object')
    missing = set(required) - set(value)
    if missing:
        raise ValueError(name + ' missing required fields: ' + ', '.join(sorted(missing)))
    unknown = set(value) - set(required) - set(optional)
    if unknown:
        raise ValueError(name + ' has unknown fields: ' + ', '.join(sorted(unknown)))
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a finite number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    return value


def _brake(value):
    value = _fields(value, 'brake', ('deceleration_mps2', 'target_speed_mps'))
    deceleration = _number(value['deceleration_mps2'], 'brake.deceleration_mps2')
    target_speed = _number(value['target_speed_mps'], 'brake.target_speed_mps')
    if deceleration <= 0:
        raise ValueError('brake.deceleration_mps2 must be greater than zero')
    if target_speed < 0:
        raise ValueError('brake.target_speed_mps must be non-negative')
    return dict(deceleration_mps2=deceleration, target_speed_mps=target_speed)


def _lane_change(value, decision_start, horizon):
    value = _fields(value, 'lane_change', ('direction', 'offset_m', 'start_s', 'duration_s'))
    direction = value['direction']
    if direction not in {'left', 'right'}:
        raise ValueError('lane_change.direction must be left or right')
    offset = _number(value['offset_m'], 'lane_change.offset_m')
    start = _number(value['start_s'], 'lane_change.start_s')
    duration = _number(value['duration_s'], 'lane_change.duration_s')
    if direction == 'left' and offset <= 0 or direction == 'right' and offset >= 0:
        raise ValueError('lane_change.offset_m sign must match direction')
    if start < decision_start:
        raise ValueError('lane_change.start_s must not precede decision start_s')
    if start > horizon:
        raise ValueError('lane_change.start_s must not exceed planning_horizon_s')
    if duration <= 0:
        raise ValueError('lane_change.duration_s must be greater than zero')
    return dict(direction=direction, offset_m=offset, start_s=start, duration_s=duration)


def validate_decision(payload):
    """Return a normalized copy of a defense_decision.v1 payload or raise ValueError."""
    payload = _fields(payload, 'decision',
        ('version', 'action_type', 'start_s', 'planning_horizon_s', 'reason'),
        ('evidence', 'brake', 'lane_change'))
    if payload['version'] != DEFENSE_DECISION_VERSION:
        raise ValueError('version must be defense_decision.v1')
    action = payload['action_type']
    if action not in ACTION_TYPES:
        raise ValueError('unknown action_type')
    start = _number(payload['start_s'], 'start_s')
    horizon = _number(payload['planning_horizon_s'], 'planning_horizon_s')
    if start < 0:
        raise ValueError('start_s must be non-negative')
    if horizon <= 0:
        raise ValueError('planning_horizon_s must be greater than zero')
    if not isinstance(payload['reason'], str):
        raise ValueError('reason must be a string')
    evidence = payload.get('evidence', [])
    if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
        raise ValueError('evidence must be a list of strings')

    needs_brake = action in {'brake', 'brake_then_lane_change'}
    needs_lane = action in {'lane_change', 'brake_then_lane_change'}
    if needs_brake and 'brake' not in payload:
        raise ValueError('brake is required for ' + action)
    if needs_lane and 'lane_change' not in payload:
        raise ValueError('lane_change is required for ' + action)
    if not needs_brake and payload.get('brake') is not None:
        raise ValueError('brake must be null or omitted for ' + action)
    if not needs_lane and payload.get('lane_change') is not None:
        raise ValueError('lane_change must be null or omitted for ' + action)
    brake = _brake(payload['brake']) if needs_brake else None
    lane_change = _lane_change(payload['lane_change'], start, horizon) if needs_lane else None
    return dict(version=DEFENSE_DECISION_VERSION, action_type=action, start_s=start,
        planning_horizon_s=horizon, reason=payload['reason'], evidence=list(evidence),
        brake=brake, lane_change=lane_change)
