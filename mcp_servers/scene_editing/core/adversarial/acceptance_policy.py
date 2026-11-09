"""Shared acceptance constants for current editing entry points.

Optimization targets and official evaluator scores are separate from acceptance.
"""
POLICY_VERSION = 'reference_acceptance.v1'
OBSERVATION_END_S = 3.
MAX_CHECK_STEP_S = .1
TIME_ROUNDING_S = .001
POSE_TOLERANCE = 1e-6
ROAD_TOLERANCE_M = .05
OFFICIAL_MARGIN_M = .1
MAX_SPEED_MPS = 15.
MAX_ACCELERATION_MPS2 = 6.
MAX_REVERSE_SPEED_MPS = 2.
MAX_LATERAL_SPEED_MPS = 1.
ATTACK_MAX_LATERAL_ACCELERATION_MPS2 = 4.
ATTACK_MAX_YAW_RATE_RADPS = 1.
ATTACK_MAX_INITIAL_SPEED_ERROR_MPS = 1.
MIN_PROGRESS_FRACTION = .5
STOP_SPEED_MPS = .1
COMFORT_REFERENCE_MPS2 = 2.4
ATTACK_NEAR_MISS_MIN_GAP_M = .15
ATTACK_EVENT_MAX_GAP_M = 2.
ROAD_FOOTPRINT_FLATNESS_M = .1
ATTACK_REVERSE_TOLERANCE_MPS = .05
ATTACK_LATERAL_SLIP_MPS = .3
ATTACK_STEERING_RAD = .6
ATTACK_STEERING_RATE_RADPS = .8
ATTACK_STATIONARY_YAW_RATE_RADPS = .08


def snapshot():
    return {name.lower(): value for name, value in globals().items()
            if name.isupper() and isinstance(value, (str, float, int))}

# Adapters consume these values rather than maintaining independent defaults.
def ego_limits():
    effective = resolve()
    return dict(max_speed_mps=effective['max_speed_mps'], max_acceleration_mps2=effective['max_acceleration_mps2'],
                max_lateral_acceleration_mps2=effective['attack_max_lateral_acceleration_mps2'],
                max_yaw_rate_radps=effective['attack_max_yaw_rate_radps'], min_clearance_m=0.)


def resolve(config=None):
    """Freeze a named research profile. Recorded speed is not a limit source."""
    from copy import deepcopy
    import math
    profiles = {
        'nuplan_case_study_v1': dict(max_speed_mps=15., max_acceleration_mps2=6.,
            attack_max_lateral_acceleration_mps2=4., generator_max_speed_mps=15.),
    }
    supplied = deepcopy(config) if config is not None else {}
    name = supplied.get('profile', 'nuplan_case_study_v1')
    if name not in profiles:
        raise ValueError('Unknown acceptance profile; choose a named profile and explicit overrides')
    result = snapshot()
    result.update(profiles[name])
    result.update(profile=name, configuration_explicit=config is not None,
        limit_source='research_profile_not_vehicle_certification', road_speed_limit_mps=None,
        vehicle_capability_source='unknown', required_checks=[], allow_safe_stop=False,
        minimum_progress_fraction=None, terminal_stop_duration_s=1., history_policy='reject')
    unknown = set(supplied)-set(result)
    if unknown:
        raise ValueError('Unknown acceptance settings: '+str(sorted(unknown)))
    result.update(supplied)
    for key, value in result.items():
        if key.startswith(('max_', 'attack_max_', 'generator_max_', 'official_margin', 'road_tolerance', 'terminal_stop_')) and (type(value) not in (int,float) or not math.isfinite(value) or value < 0):
            raise ValueError('Invalid numeric setting: '+key)
    for key in ('max_check_step_s','terminal_stop_duration_s','observation_end_s'):
        if type(result[key]) not in (int,float) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError('Positive finite value required: '+key)
    if result['observation_end_s'] < 3.:
        raise ValueError('Observation history must be at least 3 seconds')
    if result['road_speed_limit_mps'] is not None and (type(result['road_speed_limit_mps']) not in (int,float) or not math.isfinite(result['road_speed_limit_mps']) or result['road_speed_limit_mps'] <= 0):
        raise ValueError('Road limit is unknown or a positive finite value')
    if result['generator_max_speed_mps'] > result['max_speed_mps']:
        raise ValueError('Generator speed range must fit the configured reference screen')
    fraction = result['minimum_progress_fraction']
    if fraction is not None and (type(fraction) not in (int,float) or not 0 <= fraction <= 1):
        raise ValueError('minimum_progress_fraction must be null or in [0,1]')
    if type(result['allow_safe_stop']) is not bool or result['history_policy'] not in ('reject','report_existing'):
        raise ValueError('Invalid task policy')
    if not isinstance(result['required_checks'], list) or any(not isinstance(v,str) for v in result['required_checks']):
        raise ValueError('required_checks must list check names')
    return result


def task_progress(times, xy, recorded_progress, effective):
    import numpy as np
    fraction = effective['minimum_progress_fraction']
    if fraction is None:
        return True, 'progress_diagnostic_only'
    traveled = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    if traveled >= fraction*recorded_progress:
        return True, 'minimum_progress_met'
    if effective['allow_safe_stop']:
        start = times[-1]-effective['terminal_stop_duration_s']
        if start >= times[0] and len(times)>1:
            # Every interval intersecting the requested final duration must be stopped.
            mask = times[1:] > start
            speeds = np.linalg.norm(np.diff(xy,axis=0),axis=1)/np.diff(times)
            if mask.any() and np.all(speeds[mask] <= effective['stop_speed_mps']):
                return True, 'terminal_stop_exception'
    return False, 'minimum_progress_not_met'
