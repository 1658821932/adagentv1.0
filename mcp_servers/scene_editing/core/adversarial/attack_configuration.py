"""Normalize attack boundaries without treating observation history as an edit lock."""
from copy import deepcopy
import math

import acceptance_policy as policy


class AttackConfigurationError(ValueError):
    code = 'editing_configuration_error'

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def same_scene_constraints(previous, current):
    """Wall budgets/output roots may change; physics and caller contracts may not."""
    transient={'attack_stage_timeout_s','pdm_probe_root','vadv2_probe_root','inherited_constructor_budget',
        'object_reselection_count','object_reselection_reason','object_selection_feedback','_ego_speed_targets',
        '_fixed_ego_history','_ego_history_revision_target','ego_history_revision_count','expert_margin_revision_count'}
    def stable(value):
        result={key:item for key,item in value.items() if key not in transient}
        result.setdefault('prepare_ego_history',True)
        result.setdefault('ego_history_target_speed_mps',None)
        return result
    return stable(previous)==stable(current)


def _resolve_policy(value):
    try:
        return policy.resolve(value)
    except (ValueError, TypeError) as error:
        raise AttackConfigurationError('Invalid acceptance_policy: '+str(error),
            setting='acceptance_policy') from error


def normalize_constraints(constraints=None):
    supplied = deepcopy(constraints or {})
    explicit_policy = supplied.get('acceptance_policy') or {}
    effective = _resolve_policy(supplied.get('acceptance_policy'))
    config = dict(official_margin_m=policy.OFFICIAL_MARGIN_M,
                  max_ego_clearance_m=policy.ATTACK_EVENT_MAX_GAP_M, min_travel_m=0.)
    config.update(supplied)
    for key in ('official_margin_m', 'observation_end_s', 'attack_prefix_lock_s'):
        if key not in supplied:
            # Archived v1 only had the old edit/history boundary.
            config[key] = (policy.edit_start(effective)
                if key == 'attack_prefix_lock_s' and effective['policy_version'] == 'reference_acceptance.v1'
                else effective[key])
            effective[key] = config[key]
            continue
        value = supplied[key]
        min_allowed = -0.5 if key == 'official_margin_m' else 0.0
        if type(value) not in (int, float) or not math.isfinite(value) or value < min_allowed:
            raise AttackConfigurationError('Invalid attack configuration: '+key,
                setting=key, task_value=value)
        if key in explicit_policy and effective[key] != value:
            raise AttackConfigurationError(
                f'Conflicting task and acceptance setting: {key} '
                f'(task={value}, acceptance={effective[key]})',
                setting=key, task_value=value, acceptance_value=effective[key])
        effective[key] = value
    if config['observation_end_s'] <= 0:
        raise AttackConfigurationError(
            'observation_end_s is positive history duration; use attack_prefix_lock_s=0 to lock only the initial attack pose',
            setting='observation_end_s', task_value=config['observation_end_s'])
    config['acceptance_policy'] = _resolve_policy(effective)
    return config
