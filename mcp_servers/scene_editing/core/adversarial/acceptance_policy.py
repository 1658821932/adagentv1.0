"""Shared acceptance constants for current editing entry points.

Optimization targets and official evaluator scores are separate from acceptance.
"""
POLICY_VERSION = 'reference_acceptance.v2'
OBSERVATION_END_S = 1.5
EDIT_START_S = 0.
ATTACK_DANGER_START_S = 1.5
# 攻击车不需要沿原路径行驶：只锁 t=0 的起点，之后自由编辑
ATTACK_PREFIX_LOCK_S = 0.
# 危险必须出现在这个区间（秒），超出即判攻击不成立
ATTACK_WINDOW_START_S = 1.5
ATTACK_WINDOW_END_S = 7.5
MAX_CHECK_STEP_S = .1
TIME_ROUNDING_S = .001
POSE_TOLERANCE = 1e-6
ROAD_TOLERANCE_M = .05
OFFICIAL_MARGIN_M = .1
MAX_SPEED_MPS = 15.
MAX_ACCELERATION_MPS2 = 6.
MAX_REVERSE_SPEED_MPS = 2.
MAX_LATERAL_SPEED_MPS = 1.
# 自车专家的横/偏航限制独立于攻击车，保持保守
MAX_LATERAL_ACCELERATION_MPS2 = 4.
MAX_YAW_RATE_RADPS = 1.
ATTACK_MAX_LATERAL_ACCELERATION_MPS2 = 6.
ATTACK_MAX_YAW_RATE_RADPS = 1.5
ATTACK_MAX_INITIAL_SPEED_ERROR_MPS = 2.
STOP_SPEED_MPS = .1
COMFORT_REFERENCE_MPS2 = 2.4
ATTACK_NEAR_MISS_MIN_GAP_M = .15
# A 2 m near miss still scores ~0.58 on PDM, so it never becomes a training sample.
# Require the attack to nearly reach the ego; only real contact or road departure
# drives the official score to zero.
ATTACK_EVENT_MAX_GAP_M = .5
ROAD_FOOTPRINT_FLATNESS_M = .1
ATTACK_REVERSE_TOLERANCE_MPS = .05
ATTACK_LATERAL_SLIP_MPS = .3
ATTACK_STEERING_RAD = .6
ATTACK_STEERING_RATE_RADPS = 1.2
ATTACK_STATIONARY_YAW_RATE_RADPS = .08


def snapshot():
    return {name.lower(): value for name, value in globals().items()
            if name.isupper() and isinstance(value, (str, float, int))}

# Adapters consume these values rather than maintaining independent defaults.
def ego_limits():
    effective = resolve()
    return dict(max_speed_mps=effective['max_speed_mps'], max_acceleration_mps2=effective['max_acceleration_mps2'],
                max_lateral_acceleration_mps2=effective['max_lateral_acceleration_mps2'],
                max_yaw_rate_radps=effective['max_yaw_rate_radps'], min_clearance_m=0.)


def edit_start(effective=None):
    """Earliest editable scene time, independent of the WE observation horizon.

    Unversioned partial numeric fixtures and archived v1 policies retain their
    historical boundary. Fresh resolved v2 policies explicitly allow t>=0.
    """
    effective=effective or {}
    return float(effective.get('edit_start_s',effective.get('observation_end_s',OBSERVATION_END_S)))


def danger_start(effective=None):
    effective=effective or {}
    return float(effective.get('attack_danger_start_s',effective.get('observation_end_s',OBSERVATION_END_S)))


def for_new_run(config=None):
    """New public editing jobs use current time semantics, even with an old scene.

    resolve() remains version-preserving for archived trajectory verification.
    """
    from copy import deepcopy
    supplied=deepcopy(config) if config is not None else {}
    if supplied.get('policy_version')=='reference_acceptance.v1':
        for key in ('policy_version','edit_start_s','attack_danger_start_s'):
            supplied.pop(key,None)
    return resolve(supplied)


def resolve(config=None):
    """Freeze a named research profile. Recorded speed is not a limit source."""
    from copy import deepcopy
    import math
    profiles = {
        'nuplan_case_study_v1': dict(max_speed_mps=15., max_acceleration_mps2=6.,
            attack_max_lateral_acceleration_mps2=6., generator_max_speed_mps=15.),
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
        minimum_progress_fraction=None, min_progress_fraction=None,
        terminal_stop_duration_s=1., history_policy='reject')
    unknown = set(supplied)-set(result)
    if unknown:
        raise ValueError('Unknown acceptance settings: '+str(sorted(unknown)))
    result.update(supplied)
    if supplied.get('policy_version')=='reference_acceptance.v1':
        result.setdefault('policy_version','reference_acceptance.v1')
        if 'edit_start_s' not in supplied:
            result['edit_start_s']=result['observation_end_s']
        if 'attack_danger_start_s' not in supplied:
            result['attack_danger_start_s']=result['observation_end_s']
    for key, value in result.items():
        min_v = -0.5 if key.startswith('official_margin') else 0
        if key.startswith(('max_', 'attack_max_', 'generator_max_', 'official_margin', 'road_tolerance', 'terminal_stop_')) and (type(value) not in (int,float) or not math.isfinite(value) or value < min_v):
            raise ValueError('Invalid numeric setting: '+key)
    for key in ('max_check_step_s','terminal_stop_duration_s','observation_end_s'):
        if type(result[key]) not in (int,float) or not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError('Positive finite value required: '+key)
    if result['policy_version']=='reference_acceptance.v1' and result['observation_end_s'] < 3.:
        raise ValueError('Observation history must be at least 3 seconds')
    if result['policy_version'] not in ('reference_acceptance.v1','reference_acceptance.v2'):
        raise ValueError('Unknown acceptance policy version')
    if type(result['edit_start_s']) not in (int,float) or not math.isfinite(result['edit_start_s']) or result['edit_start_s']<0:
        raise ValueError('edit_start_s must be a finite nonnegative scene time')
    if type(result['attack_danger_start_s']) not in (int,float) or not math.isfinite(result['attack_danger_start_s']) or result['attack_danger_start_s']<0:
        raise ValueError('attack_danger_start_s must be a finite nonnegative scene time')
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
    if supplied.get('policy_version')=='reference_acceptance.v1':
        for key in ('edit_start_s','attack_danger_start_s'):
            if key not in supplied:
                result.pop(key,None)
    return result


def target_effect_requires_maneuver(target_effect):
    """Infer a motion goal for Agent review, without adding numeric acceptance gates.

    A temporary wait followed by driving is not a terminal-stop objective. This
    text hint remains overridable by require_route_progress; it is not completion
    evidence and never changes the user's safety or road constraints.
    """
    import re
    text = (target_effect or '').strip().lower()
    if not text:
        return False
    zh_stop = r'(?<!不)(?<!勿)(?<!别)(?<!禁止)(?<!避免)(?<!无需)(?<!不要)(?:停车|刹停|停稳|停下|驻车|完全停止|保持静止)'
    en_negated_stop = bool(re.search(r'\b(?:without|avoid|never|do not|don\'t|not)\s+stop(?:ping)?\b', text))
    en_stop = False if en_negated_stop else bool(
        re.search(r'\b(?:full stop|complete stop|stop safely|come to a stop|halt|standstill|stop)\b', text))
    stops = list(re.finditer(zh_stop, text))
    if en_stop:
        stops.extend(re.finditer(r'\b(?:full stop|complete stop|stop safely|come to a stop|halt|standstill|stop)\b', text))
    has_stop_goal = bool(stops)
    maneuver = r'(?:左转|右转|掉头|转弯|转向|直行|变道|并线|超车|绕行|通过|通行|驶过|驶离|继续行驶|恢复行驶|起步|沿.{0,12}行驶|完成.{0,8}轨迹)|\b(?:left turn|right turn|u-turn|turn|proceed|continue driving|resume driving|pass through|lane change|overtake|complete the maneuver|complete the turn)\b'
    has_maneuver_goal = bool(re.search(maneuver, text))
    if has_stop_goal:
        last_stop = max(stops, key=lambda match: match.end())
        following = text[last_stop.end():]
        if re.search(maneuver, following) and re.search(r'后|再|随后|然后|继续|恢复|\b(?:then|after|continue|resume)\b', following):
            return True
        near_stop = text[max(0, last_stop.start()-20):last_stop.end()+16]
        if has_maneuver_goal and re.search(r'必要时|需要时|临时|暂时|短暂|\b(?:briefly|temporarily|if necessary|as needed)\b', near_stop):
            return True
        return False
    return has_maneuver_goal


def resolve_ego_progress_policy(acceptance, *, target_effect='', allow_safe_stop=True, require_route_progress=None):
    """Carry the task's motion intent without inventing a progress threshold."""
    resolved = dict(acceptance)
    inferred = target_effect_requires_maneuver(target_effect) if require_route_progress is None else bool(require_route_progress)
    must_progress = (not bool(allow_safe_stop)) or inferred
    resolved['allow_safe_stop'] = bool(allow_safe_stop)
    return resolved, must_progress


def is_terminal_stopped(times, xy, effective):
    import numpy as np
    if len(times) < 2:
        return False
    start = times[-1] - effective['terminal_stop_duration_s']
    if start < times[0]:
        return False
    mask = times[1:] > start
    speeds = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(times)
    return bool(mask.any() and np.all(speeds[mask] <= effective['stop_speed_mps']))


def task_progress(times, xy, recorded_progress, effective, *, baseline_stopped=False):
    import numpy as np
    fraction = effective['minimum_progress_fraction']
    traveled = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    terminal_stopped = is_terminal_stopped(times, xy, effective)
    if not effective['allow_safe_stop'] and not baseline_stopped and terminal_stopped:
        return False, 'terminal_stop_disallowed'
    if fraction is None:
        return True, 'progress_diagnostic_only'
    if traveled >= fraction * recorded_progress:
        return True, 'minimum_progress_met'
    if effective['allow_safe_stop'] and terminal_stopped:
        return True, 'terminal_stop_exception'
    return False, 'minimum_progress_not_met'
