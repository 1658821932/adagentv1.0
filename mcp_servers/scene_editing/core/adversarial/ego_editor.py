"""Immutable constructor/FOCI candidates and parent-based ego trajectory edits."""
from workflow_timing import timed_operation

import csv
import acceptance_policy as policy
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point

from ego_action_tool import assess
from ego_context_view import compact_context
from ego_trajectory_edit import apply_intent, motion_evidence, modern_edit, start_state, inherited_brake_state, apply_curve_checks
from ego_edit_evidence import TOOL_SEMANTICS, geometry_evidence, overlap_summary, model_policy, braking_diagnostic
from ego_lane_markings import check_no_cross_lines, load_no_cross_lines
from ego_road_checks import RoadGeometry, load_scene_road, load_official_drivable
from experiment_runtime import atomic_json
from run_ego_reference import (DEFAULT_TARGET_TRACK, EGO_FRONT, EGO_REAR, EGO_WIDTH,
    evaluate, polygon, progress_with_safe_stop, scene_from_frames, smoothstep)


def _track(path, times):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(times) or any(abs(float(row['time_s'])-t) > 1e-6 for row, t in zip(rows, times)):
        raise ValueError('Candidate time grid differs from the recorded scene')
    values = np.array([[float(row['ego_x_m']), float(row['ego_y_m']), float(row['ego_yaw_rad'])] for row in rows])
    if not np.isfinite(values).all():
        raise ValueError('Candidate contains non-finite poses')
    return values[:, :2], values[:, 2]


def _write_track(path, times, xy, yaw):
    with Path(path).open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame_index', 'time_s', 'ego_x_m', 'ego_y_m', 'ego_yaw_rad'])
        for i in range(len(times)):
            writer.writerow([i, times[i], *xy[i], yaw[i]])


def _signed_speed(times, xy, yaw):
    delta = np.diff(xy, axis=0); midpoint = (yaw[:-1]+yaw[1:])/2
    return np.sum(delta/np.diff(times)[:, None]*np.column_stack([np.cos(midpoint), np.sin(midpoint)]), axis=1)


def _geometry(times, xy, yaw, actors, target_track):
    clearances = []; relative = []; overlaps = []; areas = []; shift = (EGO_FRONT-EGO_REAR)/2
    for i, (position, angle, frame) in enumerate(zip(xy, yaw, actors)):
        target = next(actor for actor in frame if actor['track'] == target_track)
        center = position+shift*np.array([np.cos(angle), np.sin(angle)])
        ego = polygon(*center, angle, EGO_FRONT+EGO_REAR, EGO_WIDTH)
        target_polygon = polygon(target['x'], target['y'], target['yaw'], target['length'], target['width'])
        clearances.append(float(ego.distance(target_polygon)))
        areas.append(float(ego.intersection(target_polygon).area))
        if ego.intersects(target_polygon):
            overlaps.append(i)
        delta = np.array([target['x'], target['y']])-position; c, s = np.cos(angle), np.sin(angle)
        relative.append([float(c*delta[0]+s*delta[1]), float(-s*delta[0]+c*delta[1])])
    return dict(clearance=np.array(clearances), target_forward_left=np.array(relative), overlaps=overlaps, intersection_area=np.array(areas))


def apply_parent_edit(times, parent_xy, parent_yaw, mode, start_index, amount, duration_s):
    """Apply one global or conflict-local operation to an immutable parent trajectory."""
    if mode not in {'retime', 'local_retime', 'reverse'}:
        raise ValueError('mode must be retime, local_retime or reverse')
    if isinstance(start_index, bool) or not isinstance(start_index, int) or not 0 <= start_index < len(times)-1:
        raise ValueError('start_index must identify an observed nonterminal frame')
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount):
        raise ValueError('amount must be finite')
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) \
            or not math.isfinite(duration_s) or not 2 <= duration_s <= 8:
        raise ValueError('duration_s must be finite in [2,8]')
    start = max(3., float(times[start_index])); phase = smoothstep((times-start)/duration_s)
    if mode in {'retime', 'local_retime'}:
        if not 0 < abs(amount) <= 2:
            raise ValueError('retime amount must be nonzero in [-2,2] seconds')
        # Continuous source-time monotonicity: phase' peaks at pi (sin^2 bump) or 1.875 (smootherstep).
        bound = duration_s/(np.pi if mode == 'local_retime' else 1.875)
        if mode == 'local_retime':
            if abs(amount) >= bound:
                raise ValueError(f'local_retime requires abs(amount) < {bound:.6g}s for monotonic rise and fall')
            u = np.clip((times-start)/duration_s, 0., 1.)
            phase = np.sin(np.pi*u)**2
        elif amount >= bound:
            raise ValueError(f'positive retime requires amount < {bound:.6g}s for this duration; negative advance is allowed')
        raw_source_times = times-amount*phase
        if np.any(np.diff(raw_source_times) <= 0) or raw_source_times.min() < -1e-9:
            raise ValueError('retime must keep sampled source times monotonic and nonnegative')
        # Advancing may reach the parent end; hold that final pose rather than inventing a future one.
        source_times = np.clip(raw_source_times, times[0], times[-1])
        xy = np.column_stack([np.interp(source_times, times, parent_xy[:, k]) for k in range(2)])
        yaw = np.interp(source_times, times, parent_yaw)
        return xy, yaw
    if not 0 < amount <= 4:
        raise ValueError('reverse amount must be in (0,4] meters')
    signed = _signed_speed(times, parent_xy, parent_yaw)
    start_speed = float(signed[min(start_index, len(signed)-1)])
    if abs(start_speed) > .5:
        raise ValueError('reverse start requires parent forward speed <= 0.5 m/s')
    if start+duration_s > times[-1]+1e-6:
        raise ValueError('reverse maneuver must fit in the observed window')
    xy = parent_xy.copy(); yaw = parent_yaw.copy(); after = np.arange(len(times)) >= start_index
    heading = parent_yaw[start_index]; direction = np.array([np.cos(heading), np.sin(heading)])
    xy[after] = parent_xy[start_index]-amount*phase[after, None]*direction
    yaw[after] = heading
    return xy, yaw


def apply_point_edits(times, parent_xy, parent_yaw, points):
    """Apply smooth forward/left/yaw offsets relative to the parent at each knot."""
    if not isinstance(points, list) or not 1 <= len(points) <= 5:
        raise ValueError('Provide 1 to 5 strictly ordered edit points')
    knots = [(float(times[0]), 0., 0., 0.)]
    for point in points:
        if not isinstance(point, dict) or set(point) != {'time_s', 'forward_m', 'left_m', 'yaw_rad'}:
            raise ValueError('Each point needs time_s, forward_m, left_m and yaw_rad')
        values = [point[k] for k in ('time_s', 'forward_m', 'left_m', 'yaw_rad')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError('Edit points must contain finite numeric values')
        time_s, forward, left, yaw = map(float, values)
        # Models commonly round the final timestamp; treat a sub-millisecond
        # difference as the endpoint rather than creating a near-zero reset.
        if abs(time_s - float(times[-1])) <= .001:
            time_s = float(times[-1])
        if not knots[-1][0] < time_s <= times[-1] or abs(forward) > 2 or abs(left) > 2 or abs(yaw) > .35:
            raise ValueError('Times must increase within the window; offsets limited to 2m and yaw to 0.35rad')
        knots.append((time_s, forward, left, yaw))
    if knots[-1][0] < times[-1]:
        knots.append((float(times[-1]), 0., 0., 0.))
    offsets = np.zeros((len(times), 3))
    for first, last in zip(knots, knots[1:]):
        mask = (times >= first[0]) & (times <= last[0])
        phase = smoothstep((times[mask]-first[0])/(last[0]-first[0]))
        offsets[mask] = np.asarray(first[1:]) + phase[:, None]*(np.asarray(last[1:])-np.asarray(first[1:]))
    c, s = np.cos(parent_yaw), np.sin(parent_yaw)
    xy = parent_xy + np.column_stack((c*offsets[:, 0]-s*offsets[:, 1], s*offsets[:, 0]+c*offsets[:, 1]))
    return xy, parent_yaw+offsets[:, 2]


def apply_candidate_repair(times, parent_xy, parent_yaw, target_xy, target_yaw, ratio, window_s=None):
    """Interpolate a proposal toward a target while retaining the initial pose."""
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('ratio must be finite in (0,1]')
    weight = float(ratio)*smoothstep((times-3.)/1.0)
    if window_s is not None:
        if not isinstance(window_s, (list, tuple)) or len(window_s) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in window_s):
            raise ValueError('window_s must contain two finite timestamps')
        start, end = window_s
        if not times[0] <= start < end <= times[-1]:
            raise ValueError('window_s must be ordered and within the observed window')
        transition = min(1., (end-start)/2)
        weight *= smoothstep((times-start)/transition)*smoothstep((end-times)/transition)
    angle_delta = np.arctan2(np.sin(target_yaw-parent_yaw), np.cos(target_yaw-parent_yaw))
    return parent_xy+weight[:, None]*(target_xy-parent_xy), parent_yaw+weight*angle_delta


class EgoEditor:
    def __init__(self, input_dir, constructor_run, run_dir, foci_run=None, resume=False,
                 target_track=DEFAULT_TARGET_TRACK, official_min_margin_m=policy.OFFICIAL_MARGIN_M):
        self.input_dir = Path(input_dir).resolve(); self.constructor_run = Path(constructor_run).resolve()
        self.foci_run = Path(foci_run).resolve() if foci_run is not None else None
        self.run_dir = Path(run_dir).resolve(); self.target_track = target_track
        if self.run_dir.exists():
            if not resume:
                raise FileExistsError('Refusing to overwrite an existing editor run')
        else:
            self.run_dir.mkdir(parents=True)
        frames = json.loads((self.input_dir/'metadata/frames.json').read_text())
        self.road = load_scene_road(self.input_dir, frames)
        self.no_cross_lines = load_no_cross_lines(self.input_dir/'metadata/no_cross_lines.json')
        official_path = self.input_dir/'metadata/map_drivable_official.json'
        self.road_official = load_official_drivable(official_path) if official_path.is_file() else None
        self.official_min_margin_m = official_min_margin_m
        source = json.loads((self.constructor_run/'reference_screen/summary.json').read_text())
        self.initial_pool_provenance=source.get('pool_provenance',dict(source='constructor_pool',uses_complete_future_observations=True,contains_validated_witness='unknown',agent_independent_rediscovery_established=False))
        saved_policy = source.get('acceptance_policy')
        self.effective_policy = policy.resolve(saved_policy) if saved_policy and 'profile' in saved_policy else policy.resolve()
        if not saved_policy or 'profile' not in saved_policy:
            self.effective_policy.update(official_margin_m=official_min_margin_m, allow_safe_stop=bool(source.get('allow_safe_stop', False)), minimum_progress_fraction=.5)
        self.official_min_margin_m = self.effective_policy['official_margin_m']
        self.allow_safe_stop = self.effective_policy['allow_safe_stop']
        if source.get('road_source') == 'official_union':
            if self.road_official is None:
                raise ValueError('Constructor requires the official drivable map')
            self.road = RoadGeometry(self.road_official.union, map_source=str(official_path),
                                     polygon_count=len(self.road_official.polygons))
        self.times, self.baseline_xy, self.baseline_yaw, self.actors = scene_from_frames(frames, target_track)
        self.baseline = evaluate(self.times, self.baseline_xy, self.baseline_yaw, self.actors, self.times, target_track)
        self.baseline_analysis = _geometry(self.times, self.baseline_xy, self.baseline_yaw, self.actors, target_track)
        self.baseline_progress = float(np.linalg.norm(np.diff(self.baseline_xy, axis=0), axis=1).sum())
        self.nodes = {}; self.history = []; self.current = None; self.selected_initial = None
        self.submitted = None; self.edit_count = 0
        if source['target_track'] != self.target_track:
            raise ValueError('Constructor target differs from the editor target')
        for record in source['candidates']:
            public_id = 'constructor_'+record['candidate_id'].removeprefix('candidate_')
            path = self.constructor_run/'reference_screen'/record['candidate_id']/'trajectory.csv'
            xy, yaw = _track(path, self.times); result = self._screen(xy, yaw, 'retime')
            if record.get('operation') is not None:
                from ego_trajectory_edit import verify_braking_candidate
                curve = verify_braking_candidate(self.times,self.baseline_xy,self.baseline_yaw,
                    xy,yaw,record['operation'],self.effective_policy['observation_end_s'])
                if curve != record['generation_curve']:
                    raise ValueError('Constructor curve evidence differs from recomputation')
                apply_curve_checks(result,curve)
                apply_intent(result,self.times,xy,yaw,record['operation'])
            if not math.isclose(result['min_target_clearance_m'], record['min_target_clearance_m'], abs_tol=1e-9) \
                    or result['target_overlap_indices'] != record['target_overlap_indices']:
                raise ValueError('Constructor candidate screening does not match archived metrics')
            self.nodes[public_id] = self._node(public_id, None, 'constructor', None, xy, yaw, result)
        if not self.nodes:
            raise ValueError('Constructor returned no candidates')
        self.constructor_best = max(self.nodes, key=self._rank)
        self.constructor_best_clearance = self.nodes[self.constructor_best]['result']['min_target_clearance_m']
        self.foci_best = None
        if self.foci_run is not None:
            config = json.loads((self.foci_run/'config.json').read_text())
            if Path(config['input_dir']).resolve() != self.input_dir \
                    or Path(config['parent_run']).resolve() != self.constructor_run \
                    or config['target_track'] != self.target_track:
                raise ValueError('FOCI input, constructor run or target differs from the editor inputs')
            foci_source = json.loads((self.foci_run/'summary.json').read_text())
            parent_id = 'constructor_'+foci_source['parent_candidate_id'].removeprefix('candidate_')
            if parent_id not in self.nodes:
                raise ValueError('FOCI parent is missing from the constructor candidates')
            for record in foci_source['candidates']:
                candidate_id = record['candidate_id']
                if not candidate_id.startswith('foci_'):
                    continue
                parent_id = 'constructor_'+record.get('parent_candidate_id',foci_source['parent_candidate_id']).removeprefix('candidate_')
                if parent_id not in self.nodes:
                    raise ValueError('FOCI candidate parent missing')
                path = self.foci_run/'candidates'/candidate_id/'trajectory.csv'
                xy, yaw = _track(path, self.times)
                parent = self.nodes[parent_id]
                heading_delta = math.atan2(math.sin(yaw[0]-parent['yaw'][0]),
                                           math.cos(yaw[0]-parent['yaw'][0]))
                if np.linalg.norm(xy[0]-parent['xy'][0]) > 1e-8 or abs(heading_delta) > 1e-8:
                    raise ValueError('FOCI candidate changes the parent initial pose')
                if config.get('version') in ('ego_foci_probe.v5','ego_foci_probe.v6','ego_foci_probe.v7','ego_foci_probe.v8'):
                    from run_ego_foci_probe import verify_candidate
                    verify_candidate(self.foci_run,candidate_id,input_dir=self.input_dir,target_track=self.target_track,
                        acceptance=self.effective_policy,times=self.times,actors=self.actors,
                        parent_xy=parent['xy'],parent_yaw=parent['yaw'])
                result = self._screen(xy, yaw, 'retime')
                if not math.isclose(result['min_target_clearance_m'], record['min_target_clearance_m'], abs_tol=1e-9) \
                        or result['target_overlap_indices'] != record['target_overlap_indices'] \
                        or (self.no_cross_lines is None and result['status'] != record['status']):
                    raise ValueError('FOCI candidate screening does not match archived metrics')
                operation = {'method':'foci_gaussian_overlap', 'weights':{
                    key:record['optimization'][key] for key in
                    ('collision_weight', 'deviation_weight', 'jerk_weight', 'lateral_weight')}}
                self.nodes[candidate_id] = self._node(candidate_id, parent_id, 'foci', operation, xy, yaw, result)
            foci_nodes = [key for key, node in self.nodes.items() if node['source'] == 'foci']
            if not foci_nodes:
                raise ValueError('FOCI returned no optimization candidates')
            self.foci_best = max(foci_nodes, key=self._rank)
        self.pool_best = max(self.nodes, key=self._rank); self.best = self.pool_best
        self.pool_best_clearance = self.nodes[self.pool_best]['result']['min_target_clearance_m']
        if resume:
            self._restore()
        else:
            for candidate_id, node in self.nodes.items():
                folder = self.run_dir/'candidates'/candidate_id; folder.mkdir(parents=True)
                _write_track(folder/'trajectory.csv', self.times, node['xy'], node['yaw'])
                atomic_json(folder/'screen.json', dict(node['result'], candidate_id=candidate_id))
                atomic_json(folder/'initial-candidate.json', dict(candidate_id=candidate_id,
                    parent_candidate_id=node['parent_id'], source=node['source'], operation=node['operation']))
            self._save_state()

    def _restore(self):
        state = json.loads((self.run_dir/'editor-state.json').read_text())
        if state.get('edit_engine_version') != 'ego_edit.v7':
            raise ValueError('Resume requires the same edit engine version')
        if state['constructor_best_candidate_id'] != self.constructor_best \
                or state.get('pool_best_candidate_id', self.constructor_best) != self.pool_best:
            raise ValueError('Archived constructor or initial-pool best differs from recomputed candidates')
        history = state['history']; count = state['edit_count']
        if len(history) != count or [row['candidate_id'] for row in history] != [
                f'edit_{i:03d}' for i in range(1, count+1)]:
            raise ValueError('Archived edit history is not contiguous')
        candidate_root = self.run_dir/'candidates'
        folders = sorted(path.name for path in candidate_root.glob('edit_*')) if candidate_root.is_dir() else []
        if folders != [f'edit_{i:03d}' for i in range(1, count+1)]:
            raise ValueError('Candidate folders differ from archived edit history')
        for index, record in enumerate(history, 1):
            candidate_id = f'edit_{index:03d}'; folder = self.run_dir/'candidates'/candidate_id
            edit = json.loads((folder/'edit.json').read_text())
            if edit['candidate_id'] != candidate_id or edit['parent_candidate_id'] != record['parent_candidate_id'] \
                    or edit['parent_candidate_id'] not in self.nodes:
                raise ValueError('Archived edit provenance mismatch')
            if edit['mode'] in {'point_edit', 'repair', 'foci'}:
                operation = {key:value for key,value in edit.items() if key not in {'candidate_id', 'parent_candidate_id'}}
                parent = self.nodes[edit['parent_candidate_id']]
                if edit['mode'] == 'point_edit':
                    expected_xy, expected_yaw, curve = self._point_edit_arrays({key:value for key,value in edit.items() if key != 'candidate_id'})
                elif edit['mode']=='foci':
                    expected_xy,expected_yaw=self._foci_arrays(edit)
                else:
                    target = self.nodes[edit['target_candidate_id']]
                    expected_xy, expected_yaw = apply_candidate_repair(self.times, parent['xy'], parent['yaw'], target['xy'], target['yaw'], edit['ratio'], edit.get('window_s'))
            else:
                operation = {key:edit[key] for key in ('mode','start_index','amount','duration_s','reason')}
                expected_xy = expected_yaw = None
            xy, yaw = _track(folder/'trajectory.csv', self.times)
            if expected_xy is not None and (not np.allclose(xy, expected_xy, atol=1e-9, rtol=0) or not np.allclose(yaw, expected_yaw, atol=1e-9, rtol=0)):
                raise ValueError('Archived edited trajectory differs from its operation')
            result = self._screen(xy, yaw, operation['mode'])
            if operation['mode'] == 'point_edit':
                apply_curve_checks(result,curve)
                apply_intent(result, self.times, xy, yaw, operation)
            result['candidate_id'] = candidate_id
            if json.loads((folder/'screen.json').read_text()) != result:
                raise ValueError('Archived edit screening differs from recomputation')
            self.nodes[candidate_id] = self._node(candidate_id, edit['parent_candidate_id'],
                'agent_edit', operation, xy, yaw, result)
        self.history = history; self.edit_count = count; self.current = state['current_candidate_id']
        self.best = state['best_candidate_id']; self.selected_initial = state['selected_initial_candidate_id']
        self.submitted = state['submitted_candidate_id']
        if (self.current is None) != (self.selected_initial is None):
            raise ValueError('Archived editor pointers disagree on the selected initial candidate')
        if self.current is not None and self.current not in self.nodes \
                or self.selected_initial is not None and self.selected_initial not in self.nodes \
                or self.best not in self.nodes:
            raise ValueError('Archived editor pointers reference unknown candidates')
        expected_best = self.pool_best
        for record in history:
            candidate_id = record['candidate_id']
            if self._rank(candidate_id) > self._rank(expected_best) or (
                    self._rank(candidate_id) == self._rank(expected_best)
                    and self._profile_comparison(candidate_id, expected_best)['dominates_reference']):
                expected_best = candidate_id
        if self.best != expected_best:
            raise ValueError('Archived best candidate differs from sequential best-candidate replay')
        if self.submitted is not None:
            accepted = json.loads((self.run_dir/'accepted-submission.json').read_text())
            if accepted['candidate_id'] != self.submitted:
                raise ValueError('Archived submission identity mismatch')

    def _node(self, candidate_id, parent_id, source, operation, xy, yaw, result):
        constructor_id = candidate_id if source == 'constructor' else self.nodes[parent_id]['constructor_id']
        return dict(candidate_id=candidate_id, parent_id=parent_id, source=source, operation=operation,
            constructor_id=constructor_id, xy=xy, yaw=yaw, result=result,
            analysis=_geometry(self.times, xy, yaw, self.actors, self.target_track))

    @timed_operation('ego._screen', None)
    def _screen(self, xy, yaw, mode):
        result = assess(self.times, self.baseline_xy, self.baseline_yaw, self.actors, xy, yaw, mode,
            target_track=self.target_track, road=self.road, road_official=self.road_official,
            official_min_margin_m=self.official_min_margin_m, allow_safe_stop=self.allow_safe_stop, no_cross_lines=self.no_cross_lines, acceptance=self.effective_policy)
        result['status'] = 'reference_screen_passed' if all(result['checks'].values()) else 'reference_screen_failed'
        return result

    def _rank(self, candidate_id):
        node = self.nodes[candidate_id]
        result = node['result']
        non_target = all(value for key,value in result['checks'].items()
            if key not in {'target_clearance','interpolated_contacts'})
        if any(contact['track'] != self.target_track for contact in result.get('interpolated_contacts',[])):
            non_target = False
        # Compare spatial paths, not equal-time poses: an in-lane stop should not
        # look like a large deviation merely because it lags behind the recording.
        deviation = 0.0
        if 'xy' in node and hasattr(self, 'baseline_xy'):
            path = LineString(self.baseline_xy)
            deviation = max(path.distance(Point(point)) for point in node['xy'])
        area = node['analysis'].get('intersection_area', np.zeros(len(self.times)))
        exposure = float(np.trapz(area,self.times))
        return (result['status'] == 'reference_screen_passed', non_target,
            not result['target_overlap_indices'], -round(float(max(area)),6), -round(exposure,6), -round(deviation,6))

    def _profile_comparison(self, candidate_id, reference_id):
        candidate = self.nodes[candidate_id]['analysis']['clearance']
        reference = self.nodes[reference_id]['analysis']['clearance']
        delta = candidate-reference
        improved = np.flatnonzero(delta > 1e-9).tolist()
        worsened = np.flatnonzero(delta < -1e-9).tolist()
        dominates = bool(improved and not worsened)
        dominated = bool(worsened and not improved)
        change = ('clearance_curve_improvement' if dominates else 'clearance_curve_regression' if dominated
                  else 'tradeoff' if improved and worsened else 'unchanged')
        def rows(indices):
            return [dict(frame_index=i, time_s=round(float(self.times[i]), 6),
                clearance_delta_m=round(float(delta[i]), 9)) for i in indices]
        candidate_motion = motion_evidence(self.times, self.nodes[candidate_id]['xy'], self.nodes[candidate_id]['yaw'])
        reference_motion = motion_evidence(self.times, self.nodes[reference_id]['xy'], self.nodes[reference_id]['yaw'])
        candidate_area = self.nodes[candidate_id]['analysis'].get('intersection_area',np.zeros(len(self.times)))
        reference_area = self.nodes[reference_id]['analysis'].get('intersection_area',np.zeros(len(self.times)))
        return dict(overlap_exposure_change_m2s=float(np.trapz(candidate_area-reference_area,self.times)),
            max_overlap_area_change_m2=float(max(candidate_area)-max(reference_area)),
            comparison_scope='Target clearance and sampled intersection area, not physical risk or overall Pareto dominance',
            max_acceleration_change_mps2=self.nodes[candidate_id]['result']['max_acceleration_mps2']-self.nodes[reference_id]['result']['max_acceleration_mps2'],
            reverse_distance_change_m=candidate_motion['reverse_distance_m']-reference_motion['reverse_distance_m'],
            profile_change=change, dominates_reference=dominates,
            dominated_by_reference=dominated, has_any_improvement=bool(improved),
            has_any_regression=bool(worsened), improved_frames=rows(improved),
            worsened_frames=rows(worsened), total_clearance_delta_m=round(float(delta.sum()), 9),
            clearance_delta_by_frame_m=[round(float(value), 9) for value in delta])

    def _preview(self, candidate_id):
        public = self._public(candidate_id)
        public.update(near_stop_indices=self._near_stop(candidate_id),
            conflict_window=self._conflict_window(candidate_id))
        return public

    def _public(self, candidate_id):
        node = self.nodes[candidate_id]; result = node['result']; analysis = node['analysis']
        operation = node['operation']
        if operation is not None:
            operation = {key:value for key,value in operation.items() if key != 'reason'}
        minimum = int(analysis['clearance'].argmin())
        terminal_dt = float(self.times[-1] - self.times[-2])
        terminal_speed = float(np.linalg.norm(node['xy'][-1] - node['xy'][-2]) / terminal_dt)
        terminal_interval_speed = terminal_speed
        state_curve=result.get('generation_curve') or {}
        terminal_source='last_sample_interval_average'
        if state_curve.get('terminal_velocity_xy_mps') is not None:
            terminal_speed=float(np.linalg.norm(state_curve['terminal_velocity_xy_mps']))
            terminal_source='reconstructed_generation_curve'
        terminal_clearance_rate = float((analysis['clearance'][-1] - analysis['clearance'][-2]) / terminal_dt)
        other = result['min_other_clearance_m']; road = result['road']
        road_official = result.get('road_official')
        public = dict(candidate_id=candidate_id, parent_id=node['parent_id'], source=(operation or {}).get('generation_source',node['source']),
            motion_evidence={key:value for key,value in motion_evidence(self.times, node['xy'], node['yaw']).items() if key != 'signed_speed_mps'},
            motion_intent=result.get('motion_intent'), generation_curve=({key:value for key,value in result['generation_curve'].items() if key not in {'continuous_state','speed_route'}} if result.get('generation_curve') else None),
            actual_default_start_state=start_state(self.times,node['xy'],node['yaw'],self.effective_policy['observation_end_s']),
            collision_geometry=overlap_summary(self.times,analysis.get('intersection_area',np.zeros(len(self.times)))),
            operation=operation, status=result['status'],
            submit_status=result.get('submit_status'), coverage=result.get('coverage'),
            input_qualification=result.get('input_qualification'), editing_status=result.get('editing_status'),
            full_window_status=result.get('full_window_status'), acceleration_components=result.get('acceleration_components'),
            native_max_acceleration_mps2=result.get('native_max_acceleration_mps2',result['max_acceleration_mps2']),
            min_target_clearance_m=round(result['min_target_clearance_m'], 6),
            minimum_clearance_frame_index=minimum, minimum_clearance_time_s=round(float(self.times[minimum]), 6),
            target_forward_left_at_minimum_m=[round(value, 4) for value in analysis['target_forward_left'][minimum]],
            target_overlap_indices=result['target_overlap_indices'],
            min_other_clearance_m=None if other is None else round(other, 6),
            min_road_margin_m=None if road is None else round(road['min_margin_m'], 6),
            road_failing_obs_frames=[] if road is None else road['failing_obs_frames'],
            min_road_margin_official_m=None if road_official is None else round(road_official['worst_corner_margin_m'], 6),
            road_official_failing_frames=[] if road_official is None else sorted({v['frame'] for v in road_official['violations']}),
            max_speed_mps=round(result['max_speed_mps'], 5),
            max_acceleration_mps2=round(result['max_acceleration_mps2'], 5),
            terminal_speed_mps=round(terminal_speed, 5),
            terminal_speed_source=terminal_source,terminal_sample_interval_speed_mps=round(terminal_interval_speed,5),
            terminal_target_clearance_rate_mps=round(terminal_clearance_rate, 5),
            traveled_m=round(result['traveled_m'], 5),
            progress_ratio_of_recorded=round(result['traveled_m']/self.baseline_progress, 4),
            comfort_reference_2_40_exceeded=bool(result['max_acceleration_mps2'] > policy.COMFORT_REFERENCE_MPS2),
            progress_reason=result['progress_reason'],
            lane_markings=result.get('lane_markings', {'status': 'unverified_no_line_data'}),
            failed_checks=[key for key, value in result['checks'].items() if not value])
        if getattr(self,'unranked_candidates',False):
            from defense_candidate_view import temporal_evidence
            public.update(temporal_evidence(self.times,node['xy'],analysis['clearance'],
                self.effective_policy['observation_end_s'],self.baseline_analysis['overlaps'],
                self.effective_policy['stop_speed_mps']))
        if getattr(self,'behavior_enabled',False):
            from defense_behavior import DrivingObjective,agent_evidence
            if not hasattr(self,'driving_objective'):
                self.driving_objective=DrivingObjective(self.times,self.baseline_xy,self.actors,self.target_track,self.effective_policy)
            public['behavior_metrics']=agent_evidence(self.driving_objective.evaluate(node['xy'],node['yaw'],analysis['clearance']))
            curve=result.get('generation_curve') or {}
            public['behavior_action_summary']=dict(start_time_s=(curve.get('actual_start_state') or {}).get('time_s'),
                exact_stop_time_s=curve.get('hold_starts_at_s'),sampled_terminal_speed_mps=terminal_interval_speed,terminal_speed_mps=terminal_speed,terminal_speed_source=terminal_source)
        return public

    def _conflict_window(self, candidate_id, parent_id=None):
        node = self.nodes[candidate_id]; analysis = node['analysis']; parent = self.nodes.get(parent_id)
        minimum = int(analysis['clearance'].argmin()); focus = {minimum, *analysis['overlaps']}
        indices = sorted({j for i in focus for j in (i-1, i, i+1) if 0 <= j < len(self.times)})
        rows = []
        for i in indices:
            row = dict(frame_index=i, time_s=round(float(self.times[i]), 6),
                candidate_clearance_m=round(float(analysis['clearance'][i]), 6),
                candidate_target_forward_left_m=[round(value, 4) for value in analysis['target_forward_left'][i]])
            if parent is not None:
                row.update(parent_clearance_m=round(float(parent['analysis']['clearance'][i]), 6),
                    clearance_delta_vs_parent_m=round(float(analysis['clearance'][i]-parent['analysis']['clearance'][i]), 6),
                    parent_target_forward_left_m=[round(value, 4) for value in parent['analysis']['target_forward_left'][i]])
            rows.append(row)
        return rows

    def _near_stop(self, candidate_id):
        node = self.nodes[candidate_id]
        return np.flatnonzero(np.abs(_signed_speed(self.times, node['xy'], node['yaw'])) <= .5).tolist()

    def _parameter_hints(self, candidate_id, mode, duration_s):
        hints = {'duration_s_range':[2, 8], 'observed_window_s':[float(self.times[0]), float(self.times[-1])]}
        if mode in {'retime', 'local_retime'}:
            hints.update(amount_range=[-2, 2], zero_amount_allowed=False,
                negative_advance='allowed; clips to and holds the parent final pose if necessary',
                local_retime='zero edit outside [start_index, start_index+duration_s]; offset returns to zero at the interval end, so extend beyond the last observation to alter that final pose')
            if isinstance(duration_s, (int, float)) and not isinstance(duration_s, bool):
                if mode == 'local_retime':
                    hints['absolute_amount_strict_upper_bound'] = duration_s/np.pi
                else:
                    hints['positive_amount_strict_upper_bound'] = duration_s/1.875
        if mode == 'reverse':
            duration = duration_s if isinstance(duration_s, (int, float)) and not isinstance(duration_s, bool) else 2
            hints.update(amount_range='(0,4] meters',
                valid_start_indices=[i for i in self._near_stop(candidate_id) if self.times[i]+duration <= self.times[-1]+1e-6])
        return hints

    def _save_state(self):
        constructor_profile = self._profile_comparison(self.best, self.constructor_best)
        pool_profile = self._profile_comparison(self.best, self.pool_best)
        best_clearance = self.nodes[self.best]['result']['min_target_clearance_m']
        atomic_json(self.run_dir/'editor-state.json', dict(edit_engine_version='ego_edit.v7', current_candidate_id=self.current,
            best_candidate_id=self.best, constructor_best_candidate_id=self.constructor_best,
            pool_best_candidate_id=self.pool_best, pool_best_clearance_m=self.pool_best_clearance,
            foci_best_candidate_id=self.foci_best,
            selected_initial_candidate_id=self.selected_initial,
            submitted_candidate_id=self.submitted, edit_count=self.edit_count, history=self.history,
            agent_improvement_over_constructor_m=best_clearance-self.constructor_best_clearance,
            agent_pareto_improvement_over_constructor=constructor_profile['dominates_reference'],
            best_profile_vs_constructor=constructor_profile,
            foci_improvement_over_constructor_m=self.pool_best_clearance-self.constructor_best_clearance,
            agent_improvement_over_pool_m=best_clearance-self.pool_best_clearance,
            agent_pareto_improvement_over_pool=pool_profile['dominates_reference'],
            best_profile_vs_pool=pool_profile,
            physical_avoidance='undetermined', carla_executed=False))

    @timed_operation('ego.select_constructor', None)
    def select_constructor(self, arguments):
        if self.current is not None:
            return dict(status='invalid', error='An initial candidate was already selected')
        candidate_id = arguments.get('candidate_id'); reason = arguments.get('reason')
        if candidate_id not in self.nodes or self.nodes[candidate_id]['source'] not in {'constructor', 'foci'} \
                or not isinstance(reason, str) or not reason.strip():
            return dict(status='invalid', error='Require a constructor or FOCI candidate_id and nonempty reason')
        self.current = candidate_id; self.selected_initial = candidate_id; self._save_state()
        return dict(status='selected', current=self._preview(candidate_id), best=self._preview(self.best),
            next_action='Edit a parent, or submit only a passing candidate; other initial candidates remain available as parents')

    def _reference_arrays(self,reference_id):
        if reference_id=='recorded_baseline':
            return self.baseline_xy,self.baseline_yaw
        if reference_id not in self.nodes:
            raise ValueError('Unknown reference path '+repr(reference_id)+'; available_reference_ids='+str(['recorded_baseline',*self.nodes]))
        return self.nodes[reference_id]['xy'],self.nodes[reference_id]['yaw']

    def _point_edit_arrays(self, arguments):
        arguments=dict(arguments)
        arguments.pop('generation_source',None)
        parent = self.nodes[arguments['parent_candidate_id']]
        mode = arguments.get('edit_mode', 'delta_parent')
        if mode != 'delta_parent':
            from ego_edit_contract import validate_point_arguments
            validate_point_arguments({key:value for key,value in arguments.items() if key not in {'mode','path_geometry_version','curve_state_version','speed_schedule_version','route_edit_version'}})
        if mode == 'delta_parent':
            xy,yaw = apply_point_edits(self.times, parent['xy'], parent['yaw'], arguments['points'])
            return xy,yaw,None
        if mode not in ('speed_profile','speed_schedule','route_edit') and arguments.get('motion_intent') not in ('unrestricted', 'forward_stop', 'forward', 'reverse'):
            raise ValueError('New editing modes require explicit motion_intent')
        boundary = self.effective_policy['observation_end_s']
        if mode in ('path_progress','set_offsets','replace_positions','brake_profile','speed_profile','speed_schedule','route_edit'):
            return modern_edit(self.times,self._reference_arrays,arguments,boundary,parent_curve=parent['result'].get('generation_curve'),acceptance=self.effective_policy)
        raise ValueError('Unknown edit_mode')

    def _foci_arrays(self, operation):
        from run_ego_foci_probe import verify_candidate
        parent=self.nodes[operation['parent_candidate_id']]
        run=Path(operation['foci_run']).resolve()
        if (self.run_dir/'generation').resolve() not in run.parents:
            raise ValueError('FOCI source must belong to this editor generation run')
        config=json.loads((run/'config.json').read_text())
        if Path(config['parent_run']).resolve()!=self.run_dir or config['parent_candidate_id']!=operation['parent_candidate_id']:
            raise ValueError('FOCI source parent identity differs')
        return verify_candidate(run,operation['foci_candidate_id'],input_dir=self.input_dir,
            target_track=self.target_track,acceptance=self.effective_policy,times=self.times,
            actors=self.actors,parent_xy=parent['xy'],parent_yaw=parent['yaw'])

    @timed_operation('ego.edit', None)
    def edit(self, arguments):
        result = self._edit(arguments)
        if arguments.get('edit_mode') in ('speed_schedule', 'route_edit'):
            from agent_tool_view import with_decision_summary
            return with_decision_summary(result, arguments)
        return result

    def _edit(self, arguments):
        if self.current is None:
            return dict(status='invalid', error='Select an initial constructor or FOCI candidate first')
        parent_id = arguments.get('parent_candidate_id')
        if parent_id not in self.nodes:
            return dict(status='invalid', error='Unknown parent_candidate_id')
        arguments=dict(arguments)
        if arguments.get('edit_mode')=='brake_profile':
            arguments.setdefault('path_geometry_version','merge_micro_knots.v1')
            arguments.setdefault('curve_state_version','continuous.v1')
        if arguments.get('edit_mode')=='speed_schedule':
            arguments.setdefault('speed_schedule_version','state_continuity.v4')
        if arguments.get('edit_mode')=='route_edit':
            arguments.setdefault('route_edit_version','state_continuity.v2')
        curve = None
        try:
            parent = self.nodes[parent_id]
            mode = arguments['mode']
            if mode == 'point_edit':
                xy, yaw, curve = self._point_edit_arrays(arguments)
            elif mode == 'foci':
                xy,yaw = self._foci_arrays(arguments)
            elif mode == 'repair':
                target = self.nodes[arguments['target_candidate_id']]
                xy, yaw = apply_candidate_repair(self.times, parent['xy'], parent['yaw'], target['xy'], target['yaw'], arguments['ratio'], arguments.get('window_s'))
            else:
                xy, yaw = apply_parent_edit(self.times, parent['xy'], parent['yaw'], mode,
                    arguments['start_index'], arguments['amount'], arguments['duration_s'])
            result = self._screen(xy, yaw, mode)
            if mode == 'point_edit':
                apply_curve_checks(result,curve)
                apply_intent(result, self.times, xy, yaw, arguments)
        except (KeyError, TypeError, ValueError) as error:
            if arguments.get('edit_mode') in ('speed_schedule','route_edit'):
                from edit_feedback import failure_feedback
                feedback=failure_feedback(error,arguments,self.effective_policy['observation_end_s'],float(self.times[-1]))
                if feedback is not None:return feedback
            message=str(error)[:500]
            hints={}
            if arguments.get('mode')!='point_edit':
                hints=self._parameter_hints(parent_id,arguments.get('mode'),arguments.get('duration_s'))
            else:
                time=arguments.get('start_time_s',arguments.get('window_s',[self.effective_policy['observation_end_s']])[0])
                if type(time) in (int,float) and math.isfinite(time):
                    hints['actual_start_state']=inherited_brake_state(self.times,parent['xy'],parent['yaw'],time,parent['result'].get('generation_curve'))
                hints.update(available_reference_ids=['recorded_baseline',*self.nodes],
                    next_action='For speed changes use edit_speed_profile. set_speed_targets supports braking, waiting, restarting and declared reverse; retime_braking preserves a stated stopping target. Keep parent route for tail repair; recorded_baseline permits checked original-route continuation. A preview spends no candidate slot. Change the deadline/start/cap indicated by the error; do not infer global infeasibility.')
            goal_failure=arguments.get('edit_mode') in ('speed_profile','speed_schedule','route_edit') and any(label in message for label in
                ('target_unreachable:','route_exhausted:','route_unavailable:','retiming_unavailable:','state_incompatible:','direction_change_requires_stop:','no extrapolation','available reference path'))
            error_summary=None
            if arguments.get('edit_mode')=='route_edit':
                error_summary='The lateral maneuver could not fit the selected route, available speed or time window. Reduce offset, allow more transition time, or explicitly choose preserve_progress; check the measured failure before retrying.'
                hints['next_action']='Use edit_route for lateral changes. Return means the spatial reference route, not the same-time parent position. To change speed first, use edit_speed_profile and then edit its candidate. No route extrapolation or silent speed changes.'
            if arguments.get('edit_mode')=='speed_schedule':
                if 'route_exhausted:' in message:
                    error_summary='The chosen route is too short for this motion. A stopped parent route ends at its stopping point; restarting does not extend it automatically.'
                    hints['retry_options']=[dict(change='Choose an earlier stop or a smaller speed/duration',condition='Keep the selected spatial route'),dict(change="Set path_source=recorded_baseline",condition='Only if continuing on the known original road route is intended; all checks still apply')]
                elif 'target_unreachable:' in message:
                    error_summary='The requested speed change is too fast for the allowed acceleration. Allow more time, start earlier, or reduce the speed change; see the required interval in error.'
                elif 'direction_change_requires_stop:' in message:
                    error_summary='Add a zero-speed target before changing between forward and reverse. Add a second zero-speed target if a waiting period is intended.'
                elif 'retiming_unavailable:' in message:
                    error_summary='This candidate has no explicit braking segment to move. Use set_speed_targets to state the desired stop/speed times.'
            return dict(status=('preview' if arguments.get('preview_only') else 'not_feasible') if goal_failure else 'invalid',
                error_summary=error_summary,
                feasible=False,candidate_created=False,error=message,
                failure_kind='requested_goal_unreachable' if goal_failure else 'invalid_edit_request',
                valid_parameter_hints=hints)
        if arguments.get('edit_mode') in ('speed_profile','speed_schedule','route_edit') and arguments.get('preview_only',False):
            return dict(status='preview',feasible=result['status']=='reference_screen_passed',candidate_created=False,reference_screen_status=result['status'],
                failed_checks=[key for key,value in result['checks'].items() if not value],
                speed_edit_summary=curve.get('speed_edit_summary'),route_edit_summary=curve.get('route_edit_summary'),actual_start_state=curve['actual_start_state'],
                min_target_clearance_m=result['min_target_clearance_m'],max_acceleration_mps2=result['max_acceleration_mps2'],
                motion_intent=result.get('motion_intent'),
                next_action='Change the request if needed; resend with preview_only=false to create a checked candidate. No candidate slot was spent.')
        self.edit_count += 1; candidate_id = f'edit_{self.edit_count:03d}'
        operation = {key: value for key, value in arguments.items() if key != 'parent_candidate_id'}
        folder = self.run_dir/'candidates'/candidate_id; folder.mkdir(parents=True, exist_ok=False)
        atomic_json(folder/'edit.json', dict(candidate_id=candidate_id, parent_candidate_id=parent_id, **operation))
        _write_track(folder/'trajectory.csv', self.times, xy, yaw)
        if curve is not None: atomic_json(folder/'curve-check.json',curve)
        result['candidate_id'] = candidate_id; atomic_json(folder/'screen.json', result)
        self.nodes[candidate_id] = self._node(candidate_id, parent_id, 'agent_edit', operation, xy, yaw, result)
        previous_best = self.best
        parent_clearance = parent['result']['min_target_clearance_m']
        best_clearance = self.nodes[previous_best]['result']['min_target_clearance_m']
        clearance = result['min_target_clearance_m']
        parent_profile = self._profile_comparison(candidate_id, parent_id)
        best_profile = self._profile_comparison(candidate_id, previous_best)
        candidate_rank = self._rank(candidate_id); best_rank = self._rank(previous_best)
        is_new_best = candidate_rank > best_rank or (
            candidate_rank == best_rank and best_profile['dominates_reference'])
        labels = ('eligibility','non_target_checks','no_overlap','max_intersection_area','intersection_exposure','path_deviation')
        improved_rank = next((labels[i] for i,(a,b) in enumerate(zip(candidate_rank,best_rank)) if a!=b),None)
        best_change_reason = ('improved_'+improved_rank if candidate_rank > best_rank else
            'clearance_curve_dominance' if is_new_best else 'none')
        if is_new_best:
            self.best = candidate_id
        # Keep a Pareto-improving child as the next parent even when its worst frame is unchanged.
        regressed_checks = [key for key,value in result['checks'].items()
            if key not in {'target_clearance','interpolated_contacts'}
            and not value and parent['result']['checks'].get(key,True)]
        if self._rank(parent_id)[1] and not candidate_rank[1] and not regressed_checks:
            regressed_checks.append('non_target_constraints')
        geometry_regressed = (parent_profile['max_overlap_area_change_m2']>1e-6
            or parent_profile['overlap_exposure_change_m2s']>1e-6)
        geometry_improved = (parent_profile['dominates_reference']
            or parent_profile['max_overlap_area_change_m2'] < -1e-6
            or parent_profile['overlap_exposure_change_m2s'] < -1e-6)
        productive = bool((is_new_best or (geometry_improved and not geometry_regressed)) and not regressed_checks)
        self.current = parent_id  # Editing parent is explicit; ranking must not select it.
        record = dict(candidate_id=candidate_id, parent_candidate_id=parent_id,
            constructor_id=self.nodes[candidate_id]['constructor_id'], mode=arguments['mode'],
            amount=arguments.get('amount'), min_target_clearance_m=clearance, delta_vs_parent_m=clearance-parent_clearance,
            dominates_parent=parent_profile['dominates_reference'], productive_edit=productive,
            improved_frame_indices=[row['frame_index'] for row in parent_profile['improved_frames']],
            worsened_frame_indices=[row['frame_index'] for row in parent_profile['worsened_frames']],
            status=result['status'])
        self.history.append(record)
        branch_history = [row for row in self.history if row['constructor_id'] == record['constructor_id']]
        consecutive_non_improving = 0
        for row in reversed(branch_history):
            if row['productive_edit']:
                break
            consecutive_non_improving += 1
        self._save_state()
        untried = [key for key, node in self.nodes.items() if node['source'] in {'constructor', 'foci'}
                   and key not in {row['parent_candidate_id'] for row in self.history}]
        recommended = self.current
        if consecutive_non_improving >= 2 and untried:
            recommended = max(untried, key=self._rank)
        return dict(status=result['status'], candidate_id=candidate_id, parent_candidate_id=parent_id,
            speed_edit_summary=None if curve is None else curve.get('speed_edit_summary'),
            route_edit_summary=None if curve is None else curve.get('route_edit_summary'),
            actual_edit=dict(parent_candidate_id=parent_id, anchor_candidate_id=arguments.get('anchor_candidate_id'),
                actual_start_state=None if curve is None else curve['actual_start_state'],
                end_xy_change_vs_parent_m=(xy[-1]-parent['xy'][-1]).tolist(),
                max_position_change_vs_parent_m=float(np.linalg.norm(xy-parent['xy'],axis=1).max())),
            min_target_clearance_m=clearance, failed_checks=[key for key, value in result['checks'].items() if not value],
            target_overlap_indices=result['target_overlap_indices'], delta_vs_parent_m=clearance-parent_clearance,
            delta_vs_previous_best_m=clearance-best_clearance, is_new_best=is_new_best,
            best_change_reason=best_change_reason, dominates_parent=parent_profile['dominates_reference'],
            productive_edit=productive, new_constraint_failures=regressed_checks,
            geometry_tradeoff=bool(geometry_improved and geometry_regressed),profile_vs_parent=parent_profile,
            profile_vs_previous_best=best_profile,
            candidate=self._public(candidate_id), current=self._public(self.current), best=self._public(self.best),
            conflict_window=self._conflict_window(candidate_id, parent_id),
            recommended_next_parent_id=recommended, recommended_parent_preview=self._preview(recommended),
            untried_initial_candidate_ids=untried,
            consecutive_non_improving_edits_on_branch=consecutive_non_improving,
            next_action='Read failed checks, actual start state, curve limits and geometric overlap; compare supported alternatives or finish with remaining uncertainties. Submit only an eligible candidate.')

    @timed_operation('ego.submit', None)
    def submit(self, arguments):
        candidate_id = arguments.get('candidate_id'); reason = arguments.get('reason')
        if candidate_id not in self.nodes or not isinstance(reason, str) or not reason.strip():
            return dict(status='invalid', error='Require an existing candidate_id and nonempty reason')
        if self.submitted is not None or (self.run_dir/'accepted-submission.json').exists():
            return dict(status='rejected', reason='A candidate was already submitted')
        if self.nodes[candidate_id]['result'].get('submit_status') == 'incomplete':
            return dict(status='rejected', reason='Required evidence is missing', missing_checks=self.nodes[candidate_id]['result']['missing_required_checks'])
        if self.nodes[candidate_id]['result']['status'] != 'reference_screen_passed':
            return dict(status='rejected', reason='Only a reference_screen_passed candidate can be submitted')
        self.submitted = candidate_id
        atomic_json(self.run_dir/'accepted-submission.json', dict(candidate_id=candidate_id,
            parent_candidate_id=self.nodes[candidate_id]['parent_id'], selection_reason=reason,
            generation_source=(self.nodes[candidate_id]['operation'] or {}).get('generation_source',self.nodes[candidate_id]['source']),
            pipeline_status='reference_screen_passed', submit_status=self.nodes[candidate_id]['result'].get('submit_status'),
            acceptance_policy=self.effective_policy, lane_compliance=self.nodes[candidate_id]['result'].get('lane_compliance', 'unknown'), physical_avoidance='undetermined', carla_executed=False))
        self._save_state(); return dict(status='accepted', candidate_id=candidate_id, physical_avoidance='undetermined')

    def model_context(self, turns_remaining, edit_candidates_remaining, last_feedback=None, pending_review_candidate_id=None):
        current = self._preview(self.current) if self.current else None
        best = self._preview(self.best)
        tried = {row['parent_candidate_id'] for row in self.history}
        untried = [key for key, node in self.nodes.items() if node['source'] in {'constructor', 'foci'} and key not in tried]
        passing = [key for key, node in self.nodes.items()
                   if node['result']['status'] == 'reference_screen_passed']
        def by_progress(key):
            return self.nodes[key]['result']['traveled_m']/self.baseline_progress
        pool_best_progress = max(passing, key=by_progress) if passing else None
        constructor_passing = [key for key in passing if self.nodes[key]['source'] == 'constructor']
        constructor_best_progress = max(constructor_passing, key=by_progress) if constructor_passing else None
        baseline_rows = []
        speed = np.linalg.norm(np.diff(self.baseline_xy, axis=0), axis=1)/np.diff(self.times)
        for i in range(len(self.times)):
            baseline_rows.append(dict(frame_index=i, time_s=round(float(self.times[i]), 6),
                recorded_ego_target_clearance_m=round(float(self.baseline_analysis['clearance'][i]), 5),
                target_forward_left_m=[round(value, 4) for value in self.baseline_analysis['target_forward_left'][i]],
                recorded_ego_speed_to_next_mps=None if i == len(self.times)-1 else round(float(speed[i]), 5)))
        context = dict(pending_review_candidate_id=pending_review_candidate_id, editor_schema_version='ego_edit.v9',tool_semantics=TOOL_SEMANTICS,initial_pool_provenance=self.initial_pool_provenance,reference_path_ids=['recorded_baseline',*self.nodes],
            reference_path_contract='recorded_baseline is the immutable recorded ego path, available to path tools but not selectable as a candidate. Other reference IDs are existing candidates. New braking/speed edits reconstruct a generated parent curve at the requested time. CSV-only parents use a reported sampled-state estimate; low-level path tools still require a matching reference. path_progress retains its sampled-state and strict matching contract.',
            ranking_scope='Numerical best ranks feasibility, maximum sampled overlap, overlap exposure and spatial deviation using 1e-6 numerical quantisation. Current is the explicit editing parent, not a ranking selection. Numerical best is not a combined comfort/progress/reverse optimum. Keep a passing fallback and compare task-relevant tradeoffs; these labels do not determine the next action.',
            acceptance_policy=model_policy(self.effective_policy), task='Choose a maneuver from the input task and scene evidence; edit when useful within the controller constraints. No maneuver is preferred universally. FOCI candidates are precomputed, not agent edits. Compare task-relevant tradeoffs and disclose evidence gaps; reference checks do not certify physical safety, complete lawfulness or unavoidability.',
            data_scope='Recorded actor observations use frame_times_s. Read acceptance_policy.max_check_step_s for configured interpolation spacing and returned road/coverage results for map checks and margins. Missing observations remain unknown; no continuous-time collision or physical certification.',
            coordinate_convention='delta_parent offsets use parent pose frames; set_offsets uses the pose of the immutable anchor trajectory at EACH target time, not the edit-start pose. Forward is positive ahead, left positive left, yaw counterclockwise in radians. replace_positions uses scene-local x/y and absolute yaw; path_progress uses signed path arc travel from its start.',
            target_track=self.target_track,
            geometry_at_edit_start=geometry_evidence(self.times,
                self.baseline_xy if self.current is None else self.nodes[self.current]['xy'],
                self.baseline_yaw if self.current is None else self.nodes[self.current]['yaw'],
                self.actors,self.target_track,self.effective_policy['observation_end_s']),
            braking_start_state=(start_state(self.times,self.baseline_xy,self.baseline_yaw,self.effective_policy['observation_end_s']) if self.current is None else inherited_brake_state(self.times,self.nodes[self.current]['xy'],self.nodes[self.current]['yaw'],self.effective_policy['observation_end_s'],self.nodes[self.current]['result'].get('generation_curve'))),
            actual_start_state=start_state(self.times,
                self.baseline_xy if self.current is None else self.nodes[self.current]['xy'],
                self.baseline_yaw if self.current is None else self.nodes[self.current]['yaw'],self.effective_policy['observation_end_s']),
            progress_policy=dict(minimum_fraction=self.effective_policy['minimum_progress_fraction'],
                allow_terminal_stop=self.allow_safe_stop, terminal_stop_duration_s=self.effective_policy['terminal_stop_duration_s'],
                note='A null minimum fraction means progress is diagnostic only; a stop does not certify safety beyond the observed horizon.'),
            observed_window_s=[float(self.times[0]), float(self.times[-1])],
            progress_edit_start=dict(time_s=self.effective_policy['observation_end_s'],
                reference_point='ego rear axle; distances are path arc travel from start, not target/body-center gap',
                recorded_xy_m=[float(np.interp(self.effective_policy['observation_end_s'],self.times,self.baseline_xy[:,axis])) for axis in range(2)],
                note='New path_progress inherits the chosen parent start velocity, then fits the requested distance/time/zero final speed. Do not add a negative offset to a stopped parent to express an earlier stop.'),
            frame_times_s=[round(float(value), 6) for value in self.times],
            recorded_scene_timeline=baseline_rows,
            goal=('Use acceptance_policy and returned check results for reference-screen requirements, '
                'observation history, motion limits and progress/stop rules. Comfort_reference_mps2 is a '
                'diagnostic preference, not an additional acceleration gate. Scene recommendations are '
                'advisory. Respect visible prohibited-crossing markings; drivable-area checks do not '
                'establish marking compliance. Missing evidence remains unknown and does not alone '
                'ban lateral movement. Compare actions against the input goal without automatically '
                'maximizing clearance, progress or score.'),
            new_edit_modes='Use edit_speed_profile for timed signed speed states or explicit braking retiming. Positive=forward, zero=stop, negative=reverse; use an explicit zero target before direction changes. Two zero targets wait, then a positive target restarts. Original braking and speed-profile operations remain available only for archive replay. Tool feedback checks actual planar acceleration including curvature. Use edit_path_progress, set_trajectory_offsets or replace_trajectory_positions; these agent tools select the backend mode and do not accept edit_mode. path_progress uses a fixed forward reference path matching the parent start pose, inherits signed start speed and accepts timed signed distance_from_start_m/speed_mps targets; only final zero-speed hold is supported, with no extrapolation or reconnect. set_offsets adds offsets to the immutable anchor trajectory pose at each target time; offsets are NOT distance travelled from the edit start. A fixed offset still follows a moving anchor until hold begins at window end; replace_positions sets scene-local absolute targets. Both position modes inherit parent start pose/velocity over window_s: hold needs the final target at window end and freezes absolute pose afterward; reconnect needs the last target before window end and returns to parent pose/velocity there. All new modes require motion_intent and end_behavior, with target times strictly after the inherited start. Start at or after acceptance_policy.observation_end_s; do not supply a start-time zero target. Intent and reverse evidence must agree. Point times normalize scene end rounding within 1ms.',
            edit_modes='Legacy delta_parent supplies 1-5 ordered time_s/forward_m/left_m/yaw_rad points; offsets add to the parent, including across repeated edits. Use a zero-offset point at acceptance_policy.observation_end_s before later offsets to preserve history. The initial offset is zero and the end returns to zero unless an endpoint point is specified. repair blends a parent toward an existing target using one ratio (0,1]; the initial pose is preserved using a 1s ramp and observation history must remain unchanged. Optional window_s restricts changes with smooth zero-offset boundaries. Each call screens one candidate.',
            baseline_min_target_clearance_m=self.baseline['min_target_clearance_m'],
            baseline_progress_m=round(self.baseline_progress, 5),
            constructor_preferred_candidate_id=self.constructor_best,
            constructor_preferred_clearance_m=self.constructor_best_clearance,
            constructor_best_by_progress_candidate_id=constructor_best_progress,
            constructor_best_by_progress_ratio=(None if constructor_best_progress is None
                else round(by_progress(constructor_best_progress), 4)),
            pool_preferred_candidate_id=self.pool_best,
            pool_preferred_clearance_m=self.pool_best_clearance,
            pool_best_by_progress_candidate_id=pool_best_progress,
            pool_best_by_progress_ratio=(None if pool_best_progress is None
                else round(by_progress(pool_best_progress), 4)),
            best_labels_note='Best can still be failed when no candidate passes. Labels compare reference evidence, do not prove lawfulness or global optimality.',
            foci_best_candidate_id=self.foci_best,
            initial_candidate_counts={source:sum(node['source'] == source for node in self.nodes.values())
                for source in ('constructor', 'foci')},
            current=current, best=best, selected_initial_candidate_id=self.selected_initial,
            current_edit_parent_id=self.current, latest_generated_candidate_id=(self.history[-1]['candidate_id'] if self.history else None),
            candidate_catalog=[self._public(key) for key in self.nodes],
            untried_initial_candidates=[self._preview(key) for key in untried],
            edit_history=self.history, last_tool_result=last_feedback,
            turns_remaining=turns_remaining, edit_candidates_remaining=edit_candidates_remaining,
            submitted_candidate_id=self.submitted,
            search_guidance='Use feedback to revise invalid calls and compare task-relevant alternatives within the remaining budget. A passing candidate is an eligible fallback, not a global optimum. Continue when a useful comparison is supported or submit with a reason; follow controller tool constraints. If no candidate can be submitted, report the search limitation.',
            context_policy='Useful conflict geometry is retained. Full trajectory arrays and prior model reasoning remain on disk but are not sent.')
        if getattr(self,'unranked_candidates',False):
            from defense_candidate_view import unranked_view, GUIDANCE
            context=unranked_view(context)
            context['candidate_order_note']='Generation order only; IDs and source do not imply quality. No recommended candidate.'
            context['comparison_guidance']=GUIDANCE
            if getattr(self,'behavior_enabled',False):
                from defense_candidate_view import BEHAVIOR_GUIDANCE
                from defense_behavior import preferences,candidate_differences
                context['behavior_preferences']=preferences()
                context['comparison_guidance']=GUIDANCE+' '+BEHAVIOR_GUIDANCE
                context['candidate_differences']=candidate_differences(context['candidate_catalog'])
                context['candidate_differences_scope']='Adjacent passed candidates in generation order, at most 10 pairs; not a ranking or exhaustive pair table. Every candidate metric remains in catalog.'
        return compact_context(context)
