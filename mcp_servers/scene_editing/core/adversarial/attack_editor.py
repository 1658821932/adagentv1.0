"""Generic target-vehicle candidate graph, editing, inspection and finalization."""
from workflow_timing import timed_operation

import csv
import json
from pathlib import Path

import acceptance_policy as policy
import numpy as np

from ego_editor import apply_candidate_repair, apply_point_edits
from ego_road_checks import load_official_drivable
from experiment_runtime import atomic_json
from run_ego_reference import EGO_FRONT, EGO_REAR, ego_path_from_frames, observed_objects_from_frames
from attack_tool_contract import candidate_eligibility, goal_diagnostics
from vehicle_trajectory import attack_time_grid, construct_trajectory, motion_heading, sample_xy, screen_attack
from displacement_tool import interpolate_offsets
from vehicle_motion import simulate_motion, check_bicycle
from attack_objective import goal_cost
from vehicle_control_search import integrate_controls, preservation_evidence
from attack_constructor import (MAX_INTERNAL_PER_CALL, MAX_INTERNAL_PER_RUN,
    goal_evidence, trial_specs, validate_goal)


def write_trajectory(path, times, xy, yaw, size):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame_index', 'time_s', 'target_x_m', 'target_y_m',
                         'target_yaw_rad', 'target_length_m', 'target_width_m', 'height_m'])
        for index, time in enumerate(times):
            writer.writerow([index, time, *xy[index], yaw[index], *size])


def read_seed(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return (np.array([float(row['time_s']) for row in rows]),
            np.array([[float(row['target_x_m']), float(row['target_y_m'])] for row in rows]),
            np.array([float(row['target_yaw_rad']) for row in rows]))


class AttackEditor:
    def __init__(self, scene, selection, run_dir, constraints):
        self.scene = Path(scene).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.selection = selection
        self.target = selection['target']
        from copy import deepcopy
        self.constraints = deepcopy(constraints)
        self.frames = json.loads((self.scene / 'metadata/frames.json').read_text())
        self.times, self.ego, self.ego_yaw = ego_path_from_frames(self.frames)
        # Decode other observations with any present track; new-object construction
        # also works in scenes with no original tracks.
        self.actors = observed_objects_from_frames(self.frames)
        if self.constraints.get('task_contract'):
            from attack_task_contract import validate_task_contract
            validate_task_contract(self.constraints['task_contract'], self.times,
                                   self.constraints.get('observation_end_s', 3.))
        seed_times, self.seed_xy, self.seed_yaw = read_seed(selection['seed_trajectory_path'])
        if seed_times.shape != self.times.shape or not np.allclose(seed_times, self.times, atol=1e-6, rtol=0):
            raise ValueError('Object-selection seed differs from the scene timestamps')
        self.road = load_official_drivable(self.scene / 'metadata/map_drivable_official.json')
        self.road_surface = None
        if self.target['operation'] == 'add_vehicle':
            from road_height import RoadHeightMap, read_recon_translation
            self.road_surface = RoadHeightMap.load(self.scene)
            self.recon_translation = read_recon_translation(self.scene)
        self.dense_times = attack_time_grid(self.times, self.constraints.get('observation_end_s', 3.))
        self.geometry_cache = {}
        self.nodes = {}
        self.inspected = set()
        self.submitted = None
        self.pdm_probe = None
        self.history = []
        self._add('seed_000', None, 'object_selection', self.seed_xy, self.seed_yaw,
                  sample_xy(self.times, self.seed_xy, self.dense_times),
                  np.interp(self.dense_times, self.times, self.seed_yaw))

    @timed_operation('attack._evaluate', None)
    def _evaluate(self, xy, yaw, dense_xy, dense_yaw, probe_surface=True, force_model=False):
        result = screen_attack(self.times, self.ego, self.ego_yaw, self.actors, self.target,
            self.seed_xy, self.seed_yaw, self.dense_times, dense_xy, dense_yaw,
            self.road, self.constraints, geometry_cache=self.geometry_cache)
        replay_xy = sample_xy(self.times, xy, self.dense_times)
        replay_yaw = np.interp(self.dense_times, self.times, np.unwrap(yaw))
        same_replay = (self.constraints.get('stationary_target') and
            np.array_equal(dense_xy, np.tile(dense_xy[0], (len(dense_xy), 1))) and
            np.array_equal(dense_yaw, np.full(len(dense_yaw), dense_yaw[0])) and
            np.array_equal(dense_xy, replay_xy) and np.array_equal(dense_yaw, replay_yaw))
        replay = result if same_replay else screen_attack(self.times, self.ego, self.ego_yaw, self.actors, self.target,
            self.seed_xy, self.seed_yaw, self.dense_times,
            sample_xy(self.times, xy, self.dense_times),
            np.interp(self.dense_times, self.times, np.unwrap(yaw)), self.road, self.constraints,
            motion_times=self.times, motion_xy=xy, geometry_cache=self.geometry_cache)
        replay_checks = dict(replay['checks'])
        result['native_replay_geometry'] = replay_checks
        result['native_replay_first_contact_time_s'] = replay['first_contact_time_s']
        result['checks']['native_replay_geometry'] = all(replay_checks.values())
        surface_failures = []
        if self.road_surface is not None:
            origin = np.asarray(self.frames[0]['ego2global'], dtype=float)
            heading = np.arctan2(origin[1, 0], origin[0, 0])
            rotation = np.array([[np.cos(heading), -np.sin(heading)],
                                 [np.sin(heading), np.cos(heading)]])
            positions = xy @ rotation.T + origin[:2, 3] - self.recon_translation[:2]
            surface_indices = [0] if (np.array_equal(xy, np.tile(xy[0], (len(xy), 1))) and
                np.array_equal(yaw, np.full(len(yaw), yaw[0]))) else range(len(xy))
            result['road_surface_evaluated_frames'] = list(surface_indices)
            for index in surface_indices:
                position, angle = positions[index], yaw[index]
                try:
                    plane = self.road_surface.placement_plane(*position, *self.target['size_m'][:2], heading + angle)
                    if plane['max_corner_residual_m'] > policy.ROAD_FOOTPRINT_FLATNESS_M:
                        raise ValueError('Vehicle footprint road flatness exceeds 0.1m')
                except ValueError as error:
                    offsets = [(dx, dy) for dx in (-2., -1., -.5, 0., .5, 1., 2.)
                               for dy in (-2., -1., -.5, 0., .5, 1., 2.) if dx or dy]
                    usable = []
                    for offset in (sorted(offsets, key=lambda value: np.linalg.norm(value)) if probe_surface else []):
                        try:
                            probe = position + np.asarray(offset) @ rotation.T
                            fitted = self.road_surface.placement_plane(*probe, *self.target['size_m'][:2], heading + angle)
                            if fitted['max_corner_residual_m'] <= policy.ROAD_FOOTPRINT_FLATNESS_M:
                                usable.append(list(offset))
                        except ValueError:
                            continue
                        if len(usable) == 4:
                            break
                    surface_failures.append(dict(time_s=float(self.times[index]),
                        target_xy_m=xy[index].tolist(), error=str(error),
                        nearby_surface_only_offsets_scene_xy_m=usable))
            result['checks']['road_surface'] = not surface_failures
        result['road_surface_failures'] = surface_failures
        if force_model or self.constraints.get('vehicle_motion_model') == 'bicycle.v1':
            result['vehicle_motion'] = check_bicycle(self.dense_times,dense_xy,dense_yaw,self.target['size_m'])
            result['checks']['vehicle_motion'] = result['vehicle_motion']['passed']
        result['reference_screen_passed'] = all(result['checks'].values())
        result['failed_checks'] = [key for key, passed in result['checks'].items() if not passed]
        return result

    def _goal_result(self, result, xy, yaw, dense_xy, dense_yaw, arguments):
        ego = sample_xy(self.times, self.ego, self.dense_times)
        ego_yaw = np.interp(self.dense_times, self.times, self.ego_yaw)
        dense = goal_evidence(self.dense_times, dense_xy, ego, ego_yaw, arguments,
                              result['first_contact_time_s'], target_yaw=dense_yaw, target_size=self.target['size_m'], task_contract=self.constraints.get('task_contract'))
        replay = goal_evidence(self.dense_times, sample_xy(self.times, xy, self.dense_times),
            ego, ego_yaw, arguments, result['native_replay_first_contact_time_s'],
            target_yaw=np.interp(self.dense_times,self.times,np.unwrap(yaw)),target_size=self.target['size_m'], task_contract=self.constraints.get('task_contract'))
        return dict(goal_met=dense['goal_met'] and replay['goal_met'], dense=dense, native_replay=replay)

    def _add(self, candidate_id, parent_id, operation, xy, yaw, dense_xy, dense_yaw):
        force_model=(isinstance(operation,dict) and operation.get('tool') in ('construct_vehicle_motion', 'construct_vehicle_controls')) or (
            parent_id is not None and self.nodes[parent_id]['result'].get('vehicle_motion') is not None)
        result = self._evaluate(xy, yaw, dense_xy, dense_yaw, force_model=force_model)
        goal = (operation.get('input', {}).get('construction_goal') if isinstance(operation, dict) else None)
        if goal is None and parent_id is not None:
            goal = self.nodes[parent_id]['result'].get('construction_goal_input')
        if goal is None and self.constraints.get('task_contract'):
            goal = dict(parent_candidate_id='seed_000',
                objective=self.constraints['task_contract']['objective'],
                max_candidates=1, reason='Evaluate the fixed task directly, without trajectory optimization')
        if goal is None:
            goal = self.constraints.get('behavior_contract')
        if force_model:
            result['motion_parameters'] = operation.get('input',{}).get('parameters',{}) if operation.get('tool')=='construct_vehicle_motion' else self.nodes[parent_id]['result'].get('motion_parameters',{}) if parent_id else {}
        if isinstance(operation, dict) and operation.get('tool') == 'construct_vehicle_controls':
            result['control_sequence'] = operation['input']['controls']
        if goal is not None:
            result['construction_goal_input'] = goal
            result['construction_goal'] = self._goal_result(result, xy, yaw, dense_xy, dense_yaw, goal)
            if goal.get('preserve_outside_window'):
                anchor = self.nodes[goal['parent_candidate_id']]
                preservation = preservation_evidence(self.dense_times, dense_xy, dense_yaw,
                    anchor['dense_xy'], anchor['dense_yaw'], goal)
                result['local_preservation'] = preservation
                result['checks']['local_preservation'] = preservation['passed']
                result['reference_screen_passed'] = all(result['checks'].values())
                result['failed_checks'] = [key for key, value in result['checks'].items() if not value]
        feedback = result['geometry_feedback']
        feedback.update(placement_valid=result['reference_screen_passed'] and
            (result.get('construction_goal') is None or result['construction_goal']['goal_met']),
            checks=dict(result['checks']), failed_checks=list(result['failed_checks']),
            task_goal_met=None if result.get('construction_goal') is None else result['construction_goal']['goal_met'])
        node = dict(candidate_id=candidate_id, parent_id=parent_id, operation=operation,
                    xy=xy, yaw=yaw, dense_xy=dense_xy, dense_yaw=dense_yaw, result=result)
        self.nodes[candidate_id] = node
        folder = self.run_dir / 'candidates' / candidate_id
        write_trajectory(folder / 'trajectory.csv', self.times, xy, yaw, self.target['size_m'])
        write_trajectory(folder / 'dense_trajectory.csv', self.dense_times, dense_xy, dense_yaw, self.target['size_m'])
        atomic_json(folder / 'evaluation.json', result)
        atomic_json(folder / 'operation.json', dict(parent_candidate_id=parent_id, operation=operation))
        return self.public(candidate_id)

    def public(self, candidate_id):
        node = self.nodes[candidate_id]
        result = node['result']
        road = result['official_road']
        return dict(candidate_id=candidate_id, parent_candidate_id=node['parent_id'],
            **candidate_eligibility(result),
            geometry_feedback=result.get('geometry_feedback'),
            construction_goal=result.get('construction_goal'),vehicle_motion=result.get('vehicle_motion'),
            reference_tracking_error_m=result.get('reference_tracking_error_m'),
            edited_point_tracking=result.get('edited_point_tracking'),
            speed_edit_summary=result.get('speed_edit_summary'),route_edit_summary=result.get('route_edit_summary'),
            acceleration_peak=result.get('acceleration_peak'),
            motion_intent=result.get('motion_intent'),native_max_acceleration_mps2=result.get('native_max_acceleration_mps2'),
            generated_curve_max_acceleration_mps2=result.get('generated_curve_max_acceleration_mps2'),
            failure_diagnostics=result.get('failure_diagnostics'),
            local_preservation=result.get('local_preservation'), diagnostic=self.diagnostic(candidate_id),
            **{key:result[key] for key in ('reference_screen_passed', 'checks', 'failed_checks',
                'min_ego_clearance_m', 'event_min_clearance_m', 'closest_time_s',
                'first_contact_time_s', 'max_speed_mps', 'max_acceleration_mps2',
                'min_turn_radius_proxy_m', 'travelled_m', 'observation_prefix_max_error_m')},
            official_margin_m=road['worst_corner_margin_m'],
            motion_violations=result['motion']['violations'],
            road_violations=road.get('violations', [])[:8],
            road_surface_failures=result['road_surface_failures'][:8],
            other_contacts=result['other_contacts'][:8],
            native_replay_geometry=result['native_replay_geometry'],
            checked_actor_pairs=result['checked_actor_pairs'], missing_actor_pairs=result['missing_actor_pairs'])

    def diagnostic(self, candidate_id):
        result = self.nodes[candidate_id]['result']
        goal = result.get('construction_goal')
        failures = list(result['failed_checks'])
        reports = []
        if goal is not None:
            goal_failures, reports = goal_diagnostics(goal)
            failures += goal_failures
        suggestions = []
        if reports:
            suggestions.append('Keep the task contract fixed. Compare requested limits and actual ranges, then adjust acceleration/steering around the failing interval; use the best parent rather than restart the seed.')
        if 'sustained_aligned_relation' in failures:
            suggestions.append('Align while moving, before stopping; extending travel requires rechecking front obstacles.')
        if any(name in failures for name in ('observed_others', 'official_road', 'road_surface')):
            suggestions.append('Change the approach or timing; increasing displacement may worsen obstacle or road failures.')
        if 'local_preservation' in failures:
            suggestions.append('The integrated path did not reconnect; allow a larger edit window explicitly or reject this edit.')
        if goal is not None and goal['goal_met'] and result['reference_screen_passed']:
            suggestions = (['Reference requirements pass. Run probe_pdm and use its actual path feedback before submission.']
                if self.constraints.get('pdm_self_check') else
                ['Hard requirements pass. Inspect this candidate image, then submit if its visual behavior matches the instruction; remaining preferences do not require another search.'])
        return dict(failures=failures, measurements=reports, suggested_changes=suggestions,
                    explanation_source='Rule-based guidance; measurements come from trajectory checks')

    def construct(self, arguments):
        xy, yaw, dense_times, dense_xy, dense_yaw = construct_trajectory(
            self.times, self.seed_xy, self.seed_yaw, arguments['points'], arguments['end_velocity_mps'],
            immutable_until_s=self.constraints.get('observation_end_s', 3.))
        if not np.array_equal(dense_times, self.dense_times):
            raise ValueError('Constructor dense timestamps differ from editor timestamps')
        return self._new('construct_trajectory', arguments, None, xy, yaw, dense_xy, dense_yaw)

    def _new(self, operation, arguments, parent_id, xy, yaw, dense_xy, dense_yaw):
        if self.constraints.get('stationary_target'):
            raise ValueError('Static placement is immutable; request object reselection in a new branch')
        if self.constraints.get('require_structured_objective') and operation in ('edit_trajectory_points', 'repair_candidate','edit_speed_profile','edit_route') and not self.nodes[parent_id]['result'].get('construction_goal_input'):
            raise ValueError('Choose a parent with a declared objective from construct_attack_candidates; seed_000 has no objective to inherit')
        desired_xy = dense_xy.copy()
        projected = False
        motion_parent=parent_id is not None and self.nodes[parent_id]['result'].get('vehicle_motion') is not None
        if (self.constraints.get('vehicle_motion_model') == 'bicycle.v1' or motion_parent) and operation not in ('construct_vehicle_motion', 'construct_vehicle_controls','edit_speed_profile','edit_route'):
            xy,yaw,dense_xy,dense_yaw = simulate_motion(self.times,self.seed_xy,self.seed_yaw,self.dense_times,
                self.target['size_m'],self.nodes[parent_id]['result'].get('motion_parameters',{}) if motion_parent else {},self.ego,self.ego_yaw,
                observation_end_s=self.constraints.get('observation_end_s',3.),reference_xy=dense_xy,reference_yaw=dense_yaw, acceptance=self.constraints.get('acceptance_policy'))
            projected = True
        candidate_id = f'candidate_{len(self.history) + 1:03d}'
        result = self._add(candidate_id, parent_id, dict(tool=operation, input=arguments),
                           xy, yaw, dense_xy, dense_yaw)
        if projected:
            self.nodes[candidate_id]['result']['reference_tracking_error_m'] = float(np.linalg.norm(dense_xy-desired_xy,axis=1).max())
            atomic_json(self.run_dir/'candidates'/candidate_id/'evaluation.json',self.nodes[candidate_id]['result'])
            result=self.public(candidate_id)
        self.history.append(dict(tool=operation, **result))
        atomic_json(self.run_dir / 'candidate-ledger.json', self.history)
        return result

    def _offset_arrays(self, arguments):
        parent = self.nodes[arguments['parent_candidate_id']]
        offsets, spline = interpolate_offsets(arguments['knots'], self.times, arguments.get('endpoint_velocities'))
        xy = parent['xy'] + offsets
        dense_xy = parent['dense_xy'] + spline(self.dense_times)
        dense_yaw = motion_heading(self.dense_times, dense_xy, parent['dense_yaw'][0])
        prefix = self.dense_times <= self.constraints.get('observation_end_s', 3.)
        dense_yaw[prefix] = parent['dense_yaw'][prefix]
        yaw = np.interp(self.times, self.dense_times, dense_yaw)
        native_prefix = self.times <= self.constraints.get('observation_end_s', 3.)
        yaw[native_prefix] = parent['yaw'][native_prefix]
        return xy, yaw, dense_xy, dense_yaw

    def construct_offsets(self, arguments):
        """Replayable internal operation emitted by the active candidate search."""
        return self._new('construct_from_offsets', arguments, arguments['parent_candidate_id'],
                         *self._offset_arrays(arguments))

    def _motion_arrays(self, arguments):
        return simulate_motion(self.times,self.seed_xy,self.seed_yaw,self.dense_times,
            self.target['size_m'],arguments['parameters'],self.ego,self.ego_yaw,
            observation_end_s=self.constraints.get('observation_end_s',3.), acceptance=self.constraints.get('acceptance_policy'))

    def construct_motion(self, arguments):
        return self._new('construct_vehicle_motion',arguments,arguments['parent_candidate_id'],
                         *self._motion_arrays(arguments))

    @timed_operation('attack.construct_controls', None)
    def construct_controls(self, arguments):
        arrays, _ = integrate_controls(self.times, self.seed_xy, self.seed_yaw, self.dense_times,
            self.target['size_m'], arguments['controls'], self.constraints.get('observation_end_s', 3.), acceptance=self.constraints.get('acceptance_policy'))
        return self._new('construct_vehicle_controls', arguments, arguments['parent_candidate_id'], *arrays)

    @timed_operation('attack.construct_pool', None)
    def construct_pool(self, arguments):
        if self.constraints.get('stationary_target'):
            raise ValueError('A stationary target has no motion controls; evaluate seed_000 or request a new placement')
        if self.constraints.get('task_contract') and 'objective' not in arguments:
            raise ValueError('Task-contract runs require structured planning preferences')
        prefix = float(self.constraints.get('observation_end_s', 3.))
        validate_goal(arguments, self.times, prefix)
        if 'objective' in arguments:
            from optimized_attack_pool import construct_pool
            return construct_pool(self, arguments)
        contract=self.constraints.get('behavior_contract')
        if contract is not None:
            if 'interaction_mode' not in arguments:
                raise ValueError('This run requires a temporal behavior goal, not a single position goal')
            if contract['interaction_mode']!='either' and arguments['interaction_mode']!=contract['interaction_mode']:
                raise ValueError('Cannot change the run interaction mode')
            if arguments['entry_side']!=contract['entry_side'] or arguments['hold_duration_s']<contract['hold_duration_s']:
                raise ValueError('Cannot weaken the run entry-side or hold-duration contract')
            if arguments['time_window_s'][0]<contract['time_window_s'][0] or arguments['time_window_s'][1]>contract['time_window_s'][1]:
                raise ValueError('Cannot widen the run behavior window')
        window = self.constraints.get('event_window_s')
        if window is not None and (arguments['time_window_s'][0] < window[0] or arguments['time_window_s'][1] > window[1]):
            raise ValueError('Constructor goal window must stay inside the configured event window')
        if not np.any((self.dense_times >= arguments['time_window_s'][0]) &
                      (self.dense_times <= arguments['time_window_s'][1])):
            raise ValueError('Goal window contains no dense samples; widen it')
        parent_id = arguments['parent_candidate_id']
        parent = self.nodes[parent_id]
        budget_path = self.run_dir / 'constructor-budget.json'
        budget = json.loads(budget_path.read_text()) if budget_path.exists() else dict(evaluations=0, calls=0)
        if budget['evaluations'] >= MAX_INTERNAL_PER_RUN:
            raise ValueError('Active constructor run evaluation budget exhausted')
        initial_evaluations = budget['evaluations']
        budget['calls'] += 1
        pool_id = f"pool_{budget['calls']:03d}"
        folder = self.run_dir / 'construction_pools' / pool_id
        folder.mkdir(parents=True, exist_ok=False)
        atomic_json(folder / 'input.json', arguments)
        atomic_json(budget_path, budget)
        trials = []
        selected = []
        completed = False
        try:
            for spec in trial_specs(self.times, self.ego, self.ego_yaw, parent['xy'], arguments, prefix, search_round=budget['calls']-1,
                    feedback=trials,parent_parameters=parent['result'].get('motion_parameters')):
                if len(trials) >= MAX_INTERNAL_PER_CALL or budget['evaluations'] >= MAX_INTERNAL_PER_RUN:
                    break
                budget['evaluations'] += 1
                atomic_json(budget_path, budget)
                is_motion = spec.get('model') == 'bicycle.v1'
                operation = dict(parent_candidate_id=parent_id,construction_goal=arguments,reason=arguments['reason'])
                if is_motion:
                    operation['parameters']=spec['parameters']
                else:
                    operation.update(knots=spec['knots'],endpoint_velocities=spec['endpoint_velocities'])
                record = dict(attempt_id=f'attempt_{len(trials)+1:03d}', **spec, operation=operation)
                try:
                    xy,yaw,dense_xy,dense_yaw = self._motion_arrays(operation) if is_motion else self._offset_arrays(operation)
                    result = self._evaluate(xy, yaw, dense_xy, dense_yaw, probe_surface=False, force_model=is_motion)
                    evidence = self._goal_result(result, xy, yaw, dense_xy, dense_yaw, arguments)
                    record.update(status='evaluated', reference_screen_passed=result['reference_screen_passed'],
                        failed_checks=result['failed_checks'], goal=evidence,
                        min_ego_clearance_m=result['min_ego_clearance_m'],
                        motion_excess=sum(max(0., violation['actual'] / violation['limit'] - 1.)
                            for violation in result['motion']['violations']),
                        motion_violations=result['motion']['violations'],
                        max_acceleration_mps2=result['max_acceleration_mps2'])
                except ValueError as error:
                    record.update(status='invalid', error=str(error))
                trials.append(record)
                with (folder / 'attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            evaluated = [trial for trial in trials if trial['status'] == 'evaluated']
            ranked = sorted(evaluated, key=lambda trial: (
                0 if trial['reference_screen_passed'] and trial['goal']['goal_met'] else
                1 if trial['reference_screen_passed'] else 2 if trial['goal']['goal_met'] else 3,
                len(trial['failed_checks']), trial['motion_excess'], trial['goal']['dense']['position_error_m'],
                trial['max_acceleration_mps2']))
            paths = []
            for trial in ranked:
                is_motion=trial.get('model')=='bicycle.v1'
                xy,_,_,_=self._motion_arrays(trial['operation']) if is_motion else self._offset_arrays(trial['operation'])
                if any(np.sqrt(np.mean(np.sum((xy - previous)**2, axis=1))) < .25 for previous in paths):
                    continue
                candidate = self.construct_motion(trial['operation']) if is_motion else self.construct_offsets(trial['operation'])
                category = ('goal_candidate' if candidate['reference_screen_passed'] and candidate['construction_goal']['goal_met']
                            else 'fallback' if candidate['reference_screen_passed'] else 'repair_material')
                selected.append(dict(category=category, attempt_id=trial['attempt_id'], **candidate))
                paths.append(xy)
                atomic_json(self.run_dir / 'candidates' / candidate['candidate_id'] / 'construction-source.json',
                            dict(pool_id=pool_id, attempt_id=trial['attempt_id']))
                if len(selected) == arguments['max_candidates']:
                    break
            completed = True
        finally:
            output = dict(pool_id=pool_id, status='interrupted' if not completed else 'constructed' if any(c['category']=='goal_candidate' for c in selected)
                else 'fallback' if any(c['category']=='fallback' for c in selected) else 'no_feasible_goal',
                evaluations_spent=budget['evaluations'] - initial_evaluations, completed_trials=len(trials),
                run_evaluations_spent=budget['evaluations'],
                stop_reason=('interrupted' if not completed else 'run_budget_exhausted' if budget['evaluations'] >= MAX_INTERNAL_PER_RUN
                    else 'call_budget_exhausted' if len(trials) >= MAX_INTERNAL_PER_CALL else 'goal_found' if any(r.get('reference_screen_passed') and r.get('goal',{}).get('goal_met') for r in trials) else 'grid_exhausted'),
                candidates=selected, limits=dict(per_call=MAX_INTERNAL_PER_CALL, per_run=MAX_INTERNAL_PER_RUN),
                scope='Bounded search of original displacement families; no complete maneuver or impossibility certificate')
            atomic_json(folder / 'output.json', output)
        return output

    @timed_operation('attack.edit', None)
    def edit(self, arguments):
        for index,point in enumerate(arguments['points']):
            for name,limit in (('forward_m',2.),('left_m',2.),('yaw_rad',.35)):
                value=point.get(name)
                if isinstance(value,(int,float)) and abs(value)>limit:
                    raise ValueError(f'points[{index}].{name}={value} is outside [-{limit},{limit}]; use a bounded edit or reconstruct')
        parent_id = arguments['parent_candidate_id']
        parent = self.nodes[parent_id]
        dense_xy, dense_yaw = apply_point_edits(self.dense_times, parent['dense_xy'],
                                               parent['dense_yaw'], arguments['points'])
        result=self._new('edit_trajectory_points', arguments, parent_id,
            sample_xy(self.dense_times, dense_xy, self.times),
            np.interp(self.times, self.dense_times, dense_yaw), dense_xy, dense_yaw)
        if self.nodes[result['candidate_id']]['result'].get('reference_tracking_error_m') is not None:
            query=np.array([point['time_s'] for point in arguments['points']],float)
            desired=sample_xy(self.dense_times,dense_xy,query)
            actual=sample_xy(self.dense_times,self.nodes[result['candidate_id']]['dense_xy'],query)
            desired_yaw=np.interp(query,self.dense_times,dense_yaw)
            actual_yaw=np.interp(query,self.dense_times,self.nodes[result['candidate_id']]['dense_yaw'])
            achieved=[dict(time_s=float(t),requested_xy_m=want.tolist(),actual_xy_m=got.tolist(),
                position_error_m=float(np.linalg.norm(got-want)),requested_yaw_rad=float(want_yaw),
                actual_yaw_rad=float(got_yaw),yaw_error_rad=float(abs(np.arctan2(np.sin(got_yaw-want_yaw),np.cos(got_yaw-want_yaw)))))
                for t,want,got,want_yaw,got_yaw in zip(query,desired,actual,desired_yaw,actual_yaw)]
            self.nodes[result['candidate_id']]['result']['edited_point_tracking']=achieved
            atomic_json(self.run_dir/'candidates'/result['candidate_id']/'evaluation.json',self.nodes[result['candidate_id']]['result'])
            result=self.public(result['candidate_id'])
            self.history[-1]=dict(tool='edit_trajectory_points',**result)
            atomic_json(self.run_dir/'candidate-ledger.json',self.history)
        return result

    @timed_operation('attack.edit_motion', None)
    def edit_motion(self, arguments):
        from agent_tool_view import with_decision_summary
        return with_decision_summary(self._edit_motion(arguments), arguments)

    def _edit_motion(self, arguments):
        from ego_trajectory_edit import modern_edit
        from scipy.interpolate import PPoly
        parent_id=arguments['parent_candidate_id'];parent=self.nodes[parent_id]
        args=dict(arguments)
        if self.constraints.get('require_structured_objective') and not parent['result'].get('construction_goal_input'):
            raise ValueError('Choose a parent with a declared objective from construct_attack_candidates; seed_000 has no objective to inherit')
        if args['edit_mode']=='speed_schedule':args.setdefault('speed_schedule_version','state_continuity.v4')
        if args['edit_mode']=='route_edit':args.setdefault('route_edit_version','state_continuity.v2')
        lookup=lambda key:(self.seed_xy,self.seed_yaw) if key=='recorded_baseline' else (self.nodes[key]['xy'],self.nodes[key]['yaw'])
        try:
            xy,yaw,curve=modern_edit(self.times,lookup,args,self.constraints.get('observation_end_s',3.),
                parent_curve=parent['result'].get('generation_curve'),acceptance=self.constraints.get('acceptance_policy'),
                parent_context=dict(reference_point='target_body_center',rear_axle_offset_m=.3*float(self.target['size_m'][0]),
                    samples=(self.dense_times,parent['dense_xy'],parent['dense_yaw'])))
        except ValueError as error:
            from edit_feedback import failure_feedback
            feedback=failure_feedback(error,args,self.constraints.get('observation_end_s',3.),float(self.times[-1]))
            if feedback is not None:return feedback
            if not any(label in str(error) for label in ('route_unavailable:','route_exhausted:','target_unreachable:','retiming_unavailable:','direction_change_requires_stop:','state_incompatible:')):raise
            return dict(status='preview' if args.get('preview_only') else 'not_feasible',feasible=False,candidate_created=False,
                error=str(error),next_action='Adjust the offset/duration/speed or choose a known longer route. The original attack objective and checks still apply.')
        cont=curve['continuous_state'];spline=PPoly(np.asarray(cont['xy_coefficients']),np.asarray(cont['times_s']))
        angles=PPoly(np.asarray(cont['yaw_coefficients']),np.asarray(cont['times_s']))
        dense_xy=parent['dense_xy'].copy();dense_yaw=parent['dense_yaw'].copy();mask=self.dense_times>spline.x[0]
        dense_xy[mask]=spline(self.dense_times[mask]);dense_yaw[mask]=angles(self.dense_times[mask])
        if args.get('preview_only'):
            result=self._evaluate(xy,yaw,dense_xy,dense_yaw)
            result=self._motion_curve_checks(result,curve)
            goal=parent['result'].get('construction_goal_input')
            task=None if goal is None else self._goal_result(result,xy,yaw,dense_xy,dense_yaw,goal)
            return dict(status='preview',candidate_created=False,feasible=result['reference_screen_passed'] and (task is None or task['goal_met']),
                checks=result['checks'],failed_checks=result['failed_checks'],
                construction_goal=task,
                reference_screen_passed=result['reference_screen_passed'],
                max_speed_mps=result['max_speed_mps'],max_acceleration_mps2=result['max_acceleration_mps2'],
                native_max_acceleration_mps2=result.get('native_max_acceleration_mps2'),
                generated_curve_max_acceleration_mps2=result.get('generated_curve_max_acceleration_mps2'),
                acceleration_peak=result.get('acceleration_peak'),motion_intent=result.get('motion_intent'),
                failure_diagnostics=result.get('failure_diagnostics'),
                speed_edit_summary=curve.get('speed_edit_summary'),route_edit_summary=curve.get('route_edit_summary'))
        operation='edit_speed_profile' if args['edit_mode']=='speed_schedule' else 'edit_route'
        created=self._new(operation,args,parent_id,xy,yaw,dense_xy,dense_yaw);cid=created['candidate_id']
        result=self._motion_curve_checks(self.nodes[cid]['result'],curve)
        # Exported target poses are body centres, unlike the expert rear-axle CSV.
        curve['actual_start_state']['reference_point']='target_body_center'
        for name in ('speed_edit_summary','route_edit_summary','acceleration_peak'):
            result[name]=curve.get(name)
        result['generation_curve']=curve
        folder=self.run_dir/'candidates'/cid;atomic_json(folder/'curve-check.json',curve);atomic_json(folder/'evaluation.json',result)
        created=self.public(cid);self.history[-1]=dict(tool=operation,**created);atomic_json(self.run_dir/'candidate-ledger.json',self.history)
        return created

    def _motion_curve_checks(self,result,curve):
        effective=policy.resolve(self.constraints.get('acceptance_policy'))
        result['checks']['continuous_motion']=(curve['max_speed_mps']<=effective['max_speed_mps']+1e-8 and curve['max_acceleration_mps2']<=effective['max_acceleration_mps2']+1e-8)
        if curve.get('edit_contract_version')=='trajectory_edit.v2':
            from edit_feedback import apply_edit_contract
            apply_edit_contract(result,curve)
        result['reference_screen_passed']=all(result['checks'].values())
        result['failed_checks']=[name for name,value in result['checks'].items() if not value]
        if curve.get('edit_contract_version')=='trajectory_edit.v2':
            result['failure_diagnostics']=dict(
                continuous_peak=curve.get('acceleration_peak') if not result['checks']['continuous_motion'] else None,
                motion=(result.get('vehicle_motion') or {}).get('violations',[]),
                intent=result.get('motion_intent'),
                next_action='Use the failed check time and actual limit to change the relevant target or transition. An inherited-prefix failure requires a different parent or earlier allowed edit; no acceptance limit is changed.')
        if result.get('geometry_feedback') is not None:
            result['geometry_feedback'].update(checks=dict(result['checks']),failed_checks=list(result['failed_checks']),placement_valid=result['reference_screen_passed'] and (result.get('construction_goal') is None or result['construction_goal']['goal_met']))
            if curve.get('edit_contract_version')=='trajectory_edit.v2':
                result['geometry_feedback'].update(max_speed_mps=result['max_speed_mps'],max_acceleration_mps2=result['max_acceleration_mps2'],acceleration_peak=curve.get('acceleration_peak'),motion_intent=result.get('motion_intent'))
        return result

    @timed_operation('attack.repair', None)
    def repair(self, arguments):
        parent_id = arguments['parent_candidate_id']
        parent, target = self.nodes[parent_id], self.nodes[arguments['target_candidate_id']]
        dense_xy, dense_yaw = apply_candidate_repair(self.dense_times, parent['dense_xy'],
            parent['dense_yaw'], target['dense_xy'], target['dense_yaw'], arguments['ratio'],
            arguments.get('window_s'))
        return self._new('repair_candidate', arguments, parent_id,
            sample_xy(self.dense_times, dense_xy, self.times),
            np.interp(self.times, self.dense_times, dense_yaw), dense_xy, dense_yaw)

    @timed_operation('attack.inspect', None)
    def inspect(self, candidate_id):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as Patch
        from run_ego_reference import polygon, EGO_FRONT, EGO_REAR, EGO_WIDTH

        node = self.nodes[candidate_id]
        goal = node['result'].get('construction_goal')
        focus_time = goal['dense']['measured_time_s'] if goal is not None else node['result']['closest_time_s']
        focus_label = 'Goal-time' if goal is not None else 'Closest-time'
        index = int(np.argmin(abs(self.times - focus_time)))
        path = self.run_dir / 'inspections' / f'{candidate_id}_{len(list((self.run_dir / "inspections").glob("*.png"))) + 1:02d}.png'
        path.parent.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
        for ax in axes:
            for geometry in self.road.polygons:
                ax.add_patch(Patch(np.asarray(geometry.exterior.coords), facecolor='#edf3e8', edgecolor='#bac8b0', linewidth=.5))
            ax.plot(self.ego[:, 0], self.ego[:, 1], color='#1565c0', label='recorded ego')
            ax.plot(node['xy'][:, 0], node['xy'][:, 1], color='#d84315', label='candidate target')
            ax.plot(self.seed_xy[:, 0], self.seed_xy[:, 1], '--', color='gray', label='object seed')
            center = self.ego[index] + (EGO_FRONT - EGO_REAR) / 2 * np.array([
                np.cos(self.ego_yaw[index]), np.sin(self.ego_yaw[index])])
            for body, color in ((polygon(*center, self.ego_yaw[index], EGO_FRONT + EGO_REAR, EGO_WIDTH), '#1565c0'),
                                (polygon(*node['xy'][index], node['yaw'][index], *self.target['size_m'][:2]), '#d84315')):
                ax.add_patch(Patch(np.asarray(body.exterior.coords), fill=False, edgecolor=color, linewidth=2))
            for actor in self.actors[index]:
                if actor['track'] == self.target['track_id']:
                    continue
                body = polygon(actor['x'], actor['y'], actor['yaw'], actor['length'], actor['width'])
                ax.add_patch(Patch(np.asarray(body.exterior.coords), facecolor='gray', alpha=.3))
                ax.text(actor['x'], actor['y'], actor['track'][:6], fontsize=6, color='#555555', clip_on=True)
            ax.set_aspect('equal'); ax.grid(alpha=.2)
            ax.set_xlabel('scene-local x (m)'); ax.set_ylabel('scene-local y (m)')
        extent = np.vstack([self.ego, node['xy'], self.seed_xy])
        axes[0].set_xlim(extent[:, 0].min() - 6, extent[:, 0].max() + 6)
        axes[0].set_ylim(extent[:, 1].min() - 6, extent[:, 1].max() + 6)
        center = (self.ego[index] + node['xy'][index]) / 2
        axes[1].set_xlim(center[0] - 12, center[0] + 12)
        axes[1].set_ylim(center[1] - 10, center[1] + 10)
        axes[0].legend(fontsize=7)
        axes[0].set_title('Full reference paths')
        axes[1].set_title(f'{focus_label} neighborhood, t={self.times[index]:.3f}s')
        fig.suptitle(f'{candidate_id} | gap={node["result"]["min_ego_clearance_m"]:.3f}m | geometry, not physical execution')
        fig.tight_layout(); fig.savefig(path); plt.close(fig)
        return dict(status='inspection_ready', candidate_id=candidate_id, path=str(path),
                    focus_label=focus_label, focus_time_s=float(self.times[index]), metrics=self.public(candidate_id))

    @timed_operation('attack.evaluate_candidate', None)
    def evaluate_candidate(self, candidate_id):
        return self.public(candidate_id)

    @timed_operation('attack.submit', None)
    def submit(self, candidate_id, reason):
        if self.submitted is not None:
            raise ValueError('A candidate has already been submitted')
        if candidate_id not in self.inspected:
            raise ValueError('Inspect and receive the trajectory image before submission')
        if not self.nodes[candidate_id]['result']['reference_screen_passed']:
            raise ValueError('Submitted candidate must pass all configured reference checks')
        goal = self.nodes[candidate_id]['result'].get('construction_goal')
        declared=self.nodes[candidate_id]['result'].get('construction_goal_input',{})
        if (self.constraints.get('task_contract') or (self.constraints.get('require_structured_objective') and not self.constraints.get('stationary_target'))) and 'objective' not in declared:
            raise ValueError('Declare and pass a structured objective before submitting this run')
        if self.constraints.get('behavior_contract') and declared.get('interaction_mode')=='either':
            raise ValueError('Declare near_miss/contact through active construction before submission')
        if goal is not None and not goal['goal_met']:
            raise ValueError('Active construction goal is not met; fallback cannot be submitted as goal completion')
        if self.constraints.get('pdm_self_check'):
            if self.pdm_probe is None:
                raise ValueError('PDM self-check was enabled but its controller is unavailable')
            self.pdm_probe.require_challenge(candidate_id)
        self.submitted = candidate_id
        atomic_json(self.run_dir / 'accepted-submission.json', dict(candidate_id=candidate_id,
            reason=reason, physical_collision='undetermined', metrics=self.public(candidate_id)))
        return dict(status='accepted', candidate_id=candidate_id)

    def context(self):
        budget_path = self.run_dir / 'constructor-budget.json'
        budget = json.loads(budget_path.read_text()) if budget_path.exists() else dict(evaluations=0, calls=0)
        snapshots = []
        for index in sorted(set(np.linspace(0, len(self.times) - 1, min(6, len(self.times)), dtype=int))):
            nearby = [actor for actor in self.actors[index]
                      if actor['track'] != self.target['track_id'] and
                      min(np.linalg.norm(np.array([actor['x'], actor['y']]) - self.ego[index]),
                          np.linalg.norm(np.array([actor['x'], actor['y']]) - self.seed_xy[index])) <= 40]
            snapshots.append(dict(time_s=float(self.times[index]), actors=nearby))
        from agent_tool_view import candidate_card
        catalog=[]
        for candidate_id in self.nodes:
            public=self.public(candidate_id)
            item=candidate_card(public)
            goal=public['construction_goal']
            if goal is not None:
                item['construction_goal']=dict(goal_met=goal['goal_met'],
                    failed_checks=public['task_failed_checks'],
                    entry_time_s=goal['dense'].get('entry_time_s'),
                    sustained_duration_s=goal['dense'].get('sustained_duration_s'),
                    error=goal_cost(goal['dense']))
            catalog.append(item)
        return dict(target=self.target, constraints=self.constraints,
            pdm_self_check=self.pdm_probe.context() if self.pdm_probe is not None else dict(enabled=False),
            interaction_planning_hints=dict(
                aligned_noncontact_forward_center_distance_m=(self.target['size_m'][0]+EGO_FRONT+EGO_REAR)/2+.15,
                note='Body-center forward distance, aligned same-lane approximation. Near miss is contact-free over the entire horizon. A stationary lead on recorded ego path may require contact mode; contact mode allows later ego defense and still forbids observation contact.'),
            constructor_budget=dict(**budget, remaining_evaluations=MAX_INTERNAL_PER_RUN-budget['evaluations'],
                                    per_call_limit=MAX_INTERNAL_PER_CALL, per_run_limit=MAX_INTERNAL_PER_RUN),
            coordinate_frame='scene-local metres, x=initial ego forward, y=left; target positions are body centers',
            timeline=[dict(time_s=float(time), ego_xy_m=ego.tolist(), seed_xy_m=seed.tolist())
                      for time, ego, seed in zip(self.times, self.ego, self.seed_xy)],
            candidates=catalog,
            other_actor_snapshots=snapshots,
            image_delivered_candidates=sorted(self.inspected))
