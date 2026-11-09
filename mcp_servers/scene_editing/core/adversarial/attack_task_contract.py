"""Caller-owned task acceptance, separate from agent planning preferences."""
from copy import deepcopy

import numpy as np

from attack_objective import objective_evidence, validate_objective
from run_ego_reference import EGO_FRONT, EGO_REAR


def validate_task_contract(contract, times, prefix):
    if not isinstance(contract, dict) or 'objective' not in contract or set(contract)-{'objective', 'path_corridor', 'schema_version', 'provenance'}:
        raise ValueError('task_contract requires objective; path_corridor and provenance are optional fields')
    if not isinstance(contract['objective'], dict):
        raise ValueError('Task objective must be an object')
    version = contract.get('schema_version')
    if version != 'attack_task.v2':
        raise ValueError('Unknown attack task contract version')
    if version == 'attack_task.v2':
        provenance = contract.get('provenance', {})
        if not isinstance(provenance, dict):
            raise ValueError('Task provenance must be an object')
        fields = [('objective.'+key, value) for key, value in contract['objective'].items()]
        if contract.get('path_corridor'):
            fields += [('path_corridor.'+key, value) for key, value in contract['path_corridor'].items()]
        for name, value in fields:
            if name == 'objective.keyframes' and value == []:
                continue
            if name == 'objective.interaction_window_s' and value == [float(np.nextafter(prefix, np.inf)), float(times[-1])]:
                continue
            evidence = provenance.get(name, {})
            if (evidence.get('kind') not in ('user_requirement', 'research_policy', 'measurement_definition') or
                not all(isinstance(evidence.get(key), str) and evidence[key].strip() for key in ('quote', 'source'))):
                raise ValueError('Hard task field needs quote, source and kind: '+name)
            if evidence.get('value') != value:
                raise ValueError('Provenance must display the exact enforced value: '+name)
    objective = deepcopy(contract['objective'])
    validate_objective(dict(parent_candidate_id='seed_000', objective=objective,
                            max_candidates=1, reason='Caller task'), times, prefix)
    contract['objective'].setdefault('interaction_window_s', objective['interaction_window_s'])
    contract['objective'].setdefault('keyframes', [])
    corridor = contract.get('path_corridor')
    if corridor is None:
        return
    if not isinstance(corridor, dict) or set(corridor) != {'start_s', 'max_offset_m', 'max_heading_difference_rad'}:
        raise ValueError('path_corridor requires start_s, max_offset_m and max_heading_difference_rad')
    if any(type(v) not in (int, float) or not np.isfinite(v) for v in corridor.values()):
        raise ValueError('Path corridor limits must be finite numbers')
    if not prefix <= corridor['start_s'] < times[-1] or corridor['max_offset_m'] <= 0 or not 0 < corridor['max_heading_difference_rad'] <= np.pi:
        raise ValueError('Invalid path corridor time or geometric limits')


def corridor_errors(xy, yaw, ego, ego_yaw, corridor):
    """Nearest segment of recorded ego body-center path, not a lane-ID test."""
    center = ego + (EGO_FRONT-EGO_REAR)/2*np.column_stack([np.cos(ego_yaw), np.sin(ego_yaw)])
    segment = np.diff(center, axis=0)
    delta = xy[:, None, :] - center[None, :-1, :]
    fractions = (np.einsum('nmi,mi->nm', delta, segment)/np.maximum(np.sum(segment**2, axis=1), 1e-12)).clip(0., 1.)
    projection = center[None, :-1, :] + fractions[..., None]*segment
    distances = np.linalg.norm(xy[:, None, :] - projection, axis=2)
    nearest = distances.argmin(axis=1)
    offset = distances[np.arange(len(xy)), nearest]
    angles = np.unwrap(ego_yaw)
    reference_yaw = angles[nearest]+fractions[np.arange(len(xy)), nearest]*np.diff(angles)[nearest]
    heading = np.abs(np.arctan2(np.sin(yaw-reference_yaw), np.cos(yaw-reference_yaw)))
    errors = np.column_stack([np.maximum(offset-corridor['max_offset_m'], 0.),
                              2*np.maximum(heading-corridor['max_heading_difference_rad'], 0.)])
    return errors, offset, heading


def plan_evidence(times, xy, yaw, ego, ego_yaw, size, preferences, contract=None):
    preferred = objective_evidence(times, xy, yaw, ego, ego_yaw, size, preferences)
    if contract is None:
        return preferred
    required = objective_evidence(times, xy, yaw, ego, ego_yaw, size, contract['objective'])
    corridor = contract.get('path_corridor')
    if corridor is None:
        return dict(required, task_contract=deepcopy(contract), preferences=preferred,
                    preferences_met=preferred['goal_met'], acceptance_basis='caller_task_contract')
    errors, offset, heading = corridor_errors(xy, yaw, ego, ego_yaw, corridor)
    mask = times >= corridor['start_s']-.001
    passed = bool(np.max(errors[mask]) <= 1e-6)
    required['path_corridor'] = dict(passed=passed, max_offset_m=float(offset[mask].max()),
        max_heading_difference_rad=float(heading[mask].max()), limits=corridor,
        scope='Recorded ego path corridor proxy; no lane identity or lane-line certification')
    if not passed:
        required['failed_checks'].append('task_path_corridor')
    required['goal_met'] = not required['failed_checks']
    required['objective_cost'] += float(np.mean(errors[mask]**2))
    # Keep old common fields so candidate ranking uses hard requirement errors.
    return dict(required, task_contract=deepcopy(contract), preferences=preferred,
                preferences_met=preferred['goal_met'], acceptance_basis='caller_task_contract')
