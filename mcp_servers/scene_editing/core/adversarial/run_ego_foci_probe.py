"""Offline dynamic 2D Gaussian-overlap trajectory probe for the recorded target scene."""
import acceptance_policy as policy
from workflow_timing import timed_operation

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from shapely.geometry import Point

from ego_action_tool import assess
from ego_road_checks import (RoadGeometry, check_drivable_area_official, ego_corners_xy,
    load_official_drivable, load_scene_road)
from experiment_runtime import atomic_json
from run_ego_reference import (DEFAULT_TARGET_TRACK, EGO_FRONT, EGO_REAR, EGO_WIDTH,
    progress_with_safe_stop, scene_from_frames)


def read_track(path, times):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(times) or any(abs(float(row['time_s'])-t) > 1e-6 for row, t in zip(rows, times)):
        raise ValueError('Parent time grid differs from the recorded scene')
    values = np.array([[float(row['ego_x_m']), float(row['ego_y_m']), float(row['ego_yaw_rad'])] for row in rows])
    if not np.isfinite(values).all():
        raise ValueError('Parent contains non-finite poses')
    return values[:, :2].copy(), values[:, 2].copy()


def write_track(path, times, xy, yaw):
    with Path(path).open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame_index', 'time_s', 'ego_x_m', 'ego_y_m', 'ego_yaw_rad'])
        for i in range(len(times)):
            writer.writerow([i, times[i], *xy[i], yaw[i]])


def spline_design(times, control_count=16, degree=3):
    times = np.asarray(times, float)
    if control_count <= degree or len(times) < control_count:
        raise ValueError('Require more trajectory samples than spline control points')
    interior_count = control_count-degree-1
    interior = np.linspace(times[0], times[-1], interior_count+2)[1:-1]
    knots = np.r_[np.repeat(times[0], degree+1), interior, np.repeat(times[-1], degree+1)]
    design = BSpline.design_matrix(times, knots, degree).toarray()
    return design, knots


def vehicle_gaussians(xy, yaw, length, width, center_shift=0., parts=3):
    xy = np.asarray(xy, float); yaw = np.asarray(yaw, float)
    length = np.broadcast_to(np.asarray(length, float), (len(xy),))
    width = np.broadcast_to(np.asarray(width, float), (len(xy),))
    local = np.linspace(-1/3, 1/3, parts)[None, :]*length[:, None] + center_shift
    c = np.cos(yaw); s = np.sin(yaw)
    means = xy[:, None, :] + np.stack([local*c[:, None], local*s[:, None]], axis=2)
    rotation = np.stack([np.stack([c, -s], axis=1), np.stack([s, c], axis=1)], axis=1)
    diagonal = np.zeros((len(xy), parts, 2, 2))
    diagonal[:, :, 0, 0] = (length[:, None]/parts/2)**2
    diagonal[:, :, 1, 1] = (width[:, None]/2)**2
    covariance = rotation[:, None] @ diagonal @ np.swapaxes(rotation[:, None], -1, -2)
    return means, covariance


def target_gaussians(actors, target_track=DEFAULT_TARGET_TRACK):
    means = []; covariance = []
    for frame in actors:
        target = next(actor for actor in frame if actor['track'] == target_track)
        mean, cov = vehicle_gaussians(np.array([[target['x'], target['y']]]), np.array([target['yaw']]),
            target['length'], target['width'])
        means.append(mean[0]); covariance.append(cov[0])
    return np.asarray(means), np.asarray(covariance)


def overlap_integral(ego_means, ego_covariance, target_means, target_covariance):
    delta = ego_means[:, :, None, :]-target_means[:, None, :, :]
    covariance = ego_covariance[:, :, None]+target_covariance[:, None, :]
    inverse = np.linalg.inv(covariance)
    determinant = np.linalg.det(covariance)
    mahalanobis = np.einsum('...i,...ij,...j->...', delta, inverse, delta)
    density = np.exp(-.5*mahalanobis)/(2*np.pi*np.sqrt(determinant))
    return density.sum(axis=(1, 2))


class FociProblem:
    def __init__(self, times, parent_xy, parent_yaw, actors, control_count=16, freeze_before_s=8.,
            target_track=DEFAULT_TARGET_TRACK, official_road=None, road_margin_m=0.1,
            edit_window_s=None, observation_end_s=3., behavior_objective=None, edit_goals=None):
        self.times = np.asarray(times, float)
        self.observation_end_s = float(observation_end_s)
        self.edit_window_s = edit_window_s
        self.behavior_objective=behavior_objective
        self.edit_goals=edit_goals or {}
        if self.edit_goals.get('speed_targets'):
            from speed_profile_contract import validate_targets
            validate_targets(self.edit_goals['speed_targets'],start=edit_window_s[0],end=edit_window_s[1])
        if self.edit_goals.get('terminal_behavior')=='stop' and (edit_window_s is None or abs(edit_window_s[1]-self.times[-1])>1e-8 or edit_window_s[0]>=self.times[-3]):
            raise ValueError('Stop goal requires a terminal window covering at least the last two intervals')
        if edit_window_s is not None:
            if (len(edit_window_s)!=2 or not np.isfinite(edit_window_s).all()
                    or not observation_end_s <= edit_window_s[0] < edit_window_s[1] <= self.times[-1]+1e-8):
                raise ValueError('FOCI edit window must lie after observation and inside scene')
            span=edit_window_s[1]-edit_window_s[0]
            u=np.clip((self.times-edit_window_s[0])/min(1.,span/2),0.,1.)
            v=np.clip((edit_window_s[1]-self.times)/min(1.,span/2),0.,1.)
            smooth=lambda x:x**3*(10+x*(-15+6*x))
            self.window_weight=smooth(u)*(1. if behavior_objective is not None and abs(edit_window_s[1]-self.times[-1])<1e-8 else smooth(v))
        self.parent_xy = np.asarray(parent_xy, float); self.parent_yaw = np.asarray(parent_yaw, float)
        self.design, self.knots = spline_design(self.times, control_count)
        self.initial_control = np.linalg.lstsq(self.design, np.column_stack([parent_xy, parent_yaw]), rcond=None)[0]
        # A clamped spline starts at its first control point.
        self.initial_control[0] = np.r_[self.parent_xy[0], self.parent_yaw[0]]
        self.control_times = np.linspace(self.times[0], self.times[-1], control_count)
        self.frozen = self.control_times < freeze_before_s-1e-9
        self.frozen[0] = True
        self.target_means, self.target_covariance = target_gaussians(actors, target_track)
        self.zero_xy, self.zero_yaw = self.trajectory(np.zeros_like(self.initial_control))
        self.initial_overlap = overlap_integral(*vehicle_gaussians(self.zero_xy, self.zero_yaw,
            EGO_FRONT+EGO_REAR, EGO_WIDTH, (EGO_FRONT-EGO_REAR)/2), self.target_means, self.target_covariance)
        scale = float(self.initial_overlap.mean())
        self.overlap_scale = scale if math.isfinite(scale) and scale > 0 else None
        self.official_road = official_road
        self.road_margin_m = float(road_margin_m)
        self._road_boundary = official_road.union.boundary if official_road is not None else None

    def road_penalty(self, xy, yaw):
        """Mean squared shortfall of each body corner below road_margin_m inside the
        official drivable union (0 when every corner keeps the margin). Corners outside
        the union count a negative margin, so the penalty grows quadratically outward."""
        if self.official_road is None:
            return 0.0
        union = self.official_road.union
        boundary = self._road_boundary
        total = 0.0
        for i in range(len(self.times)):
            for corner in ego_corners_xy(xy[i, 0], xy[i, 1], yaw[i]):
                point = Point(corner)
                if union.covers(point):
                    margin = boundary.distance(point)
                else:
                    margin = -union.distance(point)
                shortfall = self.road_margin_m-margin
                if shortfall > 0:
                    total += shortfall**2
        return float(total/(4*len(self.times)))

    def trajectory(self, delta):
        delta=np.asarray(delta).reshape(self.initial_control.shape)
        if self.edit_window_s is None:
            poses = self.design @ (self.initial_control+delta)
        else:
            # Zero edit is exactly the parent, not a fresh spline fit of the parent.
            poses=np.column_stack([self.parent_xy,self.parent_yaw])+(self.design @ delta)*self.window_weight[:,None]
        if self.edit_goals.get('preserve_route',False):
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.parent_xy,axis=0),axis=1))]
            keep=np.r_[np.diff(arc)>1e-8,True]
            shifted=np.clip(arc+(self.design@delta)[:,0]*self.window_weight,0.,arc[-1])
            if self.edit_goals.get('terminal_behavior')=='stop':
                shifted=self._terminal_hold(shifted[:,None])[:,0]
            poses[:,:2]=np.column_stack([np.interp(shifted,arc[keep],self.parent_xy[keep,k]) for k in range(2)])
            poses[:,2]=np.interp(shifted,arc[keep],np.unwrap(self.parent_yaw)[keep])
            # Preserve all untouched poses exactly, including stationary duplicates.
            unchanged=(shifted==arc)
            poses[unchanged,:2]=self.parent_xy[unchanged];poses[unchanged,2]=self.parent_yaw[unchanged]
        elif self.edit_goals.get('terminal_behavior')=='stop':
            poses=self._terminal_hold(poses)
        prefix = self.times <= self.observation_end_s+1e-8
        poses[prefix, :2] = self.parent_xy[prefix]
        poses[prefix, 2] = self.parent_yaw[prefix]
        return poses[:, :2], poses[:, 2]

    def _terminal_hold(self, values):
        # Explicit sampled terminal hold, with a zero terminal derivative in the
        # preceding Hermite segment. No claim of a new continuous FOCI model.
        from scipy.interpolate import CubicHermiteSpline
        result=values.copy();begin=len(self.times)-3
        velocity=(values[begin]-values[begin-1])/(self.times[begin]-self.times[begin-1])
        spline=CubicHermiteSpline(self.times[[begin,-2]],values[[begin,-1]],np.stack([velocity,np.zeros_like(velocity)]))
        result[begin:-1]=spline(self.times[begin:-1]);result[-1]=result[-2]
        return result

    def overlap(self, xy, yaw):
        means, covariance = vehicle_gaussians(xy, yaw, EGO_FRONT+EGO_REAR, EGO_WIDTH,
            (EGO_FRONT-EGO_REAR)/2)
        return overlap_integral(means, covariance, self.target_means, self.target_covariance)

    def objective(self, flat_delta, collision_weight, deviation_weight, jerk_weight, lateral_weight,
            road_weight=0.):
        delta = flat_delta.reshape(self.initial_control.shape)
        xy, yaw = self.trajectory(delta)
        overlap = float(self.overlap(xy, yaw).mean())/self.overlap_scale if self.overlap_scale else 0.
        correction_xy = xy-self.zero_xy; correction_yaw = yaw-self.zero_yaw
        deviation = float(np.mean(correction_xy**2)+.1*np.mean(correction_yaw**2))
        jerk = float(np.mean(np.diff(correction_xy, n=3, axis=0)**2)
            +.05*np.mean(np.diff(correction_yaw, n=3)**2))
        motion = np.diff(xy, axis=0)/np.diff(self.times)[:, None]
        midpoint = (yaw[:-1]+yaw[1:])/2
        lateral = np.sum(motion*np.column_stack([-np.sin(midpoint), np.cos(midpoint)]), axis=1)
        lateral_penalty = float(np.mean(np.maximum(np.abs(lateral)-.75, 0.)**2))
        target_cost=0.
        if self.edit_goals.get('speed_targets'):
            from ego_speed_schedule import sampled_target_errors
            target_cost=sampled_target_errors(self.times,xy,yaw,self.edit_goals['speed_targets'])['mean_squared_error_mps2']
        if self.behavior_objective is not None:
            behavior=self.behavior_objective.evaluate(xy,yaw)
            # Extra separation receives no reward after the modeled reserve is met.
            emphasis=0.
            if self.edit_goals.get('improvement_goal')=='recover_progress':
                emphasis=behavior['cost_components']['progress_loss']
            elif self.edit_goals.get('improvement_goal')=='smoother_motion':
                emphasis=.08*behavior['cost_components']['acceleration']+.02*behavior['cost_components']['jerk']
            return (behavior['optimization_cost']+emphasis+target_cost+collision_weight*overlap*min(1.,behavior['reserve_shortfall_cost'])
                +.01*deviation/max(1.,self.behavior_objective.available**2)
                +lateral_weight*lateral_penalty+road_weight*self.road_penalty(xy,yaw))
        return (target_cost+collision_weight*overlap+deviation_weight*deviation+jerk_weight*jerk
            +lateral_weight*lateral_penalty+road_weight*self.road_penalty(xy, yaw))

    def bounds(self, xy_limit=2., yaw_limit=.35):
        bounds = []
        for frozen in self.frozen:
            bounds.extend([(0., 0.)]*3 if frozen else ([(-xy_limit,xy_limit),(0.,0.),(0.,0.)] if self.edit_goals.get('preserve_route') else [(-xy_limit, xy_limit), (-xy_limit, xy_limit),
                (-yaw_limit, yaw_limit)]))
        return bounds

    def optimize(self, collision_weight, deviation_weight, jerk_weight, lateral_weight, road_weight=0.,
            max_iterations=250):
        start = time.monotonic()
        initial = np.zeros(self.initial_control.size)
        result = minimize(self.objective, initial,
            args=(collision_weight, deviation_weight, jerk_weight, lateral_weight, road_weight),
            method='L-BFGS-B', bounds=self.bounds(), options={'maxiter':max_iterations, 'maxls':40, 'ftol':1e-12})
        xy, yaw = self.trajectory(result.x.reshape(self.initial_control.shape))
        return xy, yaw, dict(success=bool(result.success), solver_status=int(result.status),
            message=str(result.message), iterations=int(result.nit), function_evaluations=int(result.nfev),
            objective=float(result.fun), collision_weight=collision_weight,
            deviation_weight=deviation_weight, jerk_weight=jerk_weight, lateral_weight=lateral_weight,
            road_weight=road_weight, road_margin_m=self.road_margin_m,
            road_penalty_final=float(self.road_penalty(xy, yaw)),
            lateral_soft_target_mps=.75, initial_mean_overlap_per_frame=self.overlap_scale,
            final_mean_overlap_per_frame=float(self.overlap(xy, yaw).mean()),
            control_delta=result.x.reshape(self.initial_control.shape).tolist(),
            wall_time_s=time.monotonic()-start)


def screen(times, baseline_xy, baseline_yaw, actors, xy, yaw, target_track=DEFAULT_TARGET_TRACK, road=None,
        road_official=None, official_min_margin_m=policy.OFFICIAL_MARGIN_M, allow_safe_stop=False, no_cross_lines=None, acceptance=None):
    result = assess(times, baseline_xy, baseline_yaw, actors, xy, yaw, 'retime',
        target_track=target_track, road=road, road_official=road_official,
        official_min_margin_m=official_min_margin_m, allow_safe_stop=allow_safe_stop, no_cross_lines=no_cross_lines, acceptance=acceptance)
    return result


@timed_operation('ego.foci', 'run_dir')
def run(input_dir, parent_run, candidate_id, run_dir, max_iterations=250, target_track=DEFAULT_TARGET_TRACK,
        freeze_before_s=None, road_margin_target_m=0.15, edit_window_s=None, max_candidates=None, additional_parents=None, preserve_route=False, terminal_behavior='free', improvement_goal='balanced', speed_targets=None):
    input_dir = Path(input_dir).resolve(); parent_run = Path(parent_run).resolve(); run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError('Refusing to overwrite an existing FOCI probe run')
    run_dir.mkdir(parents=True)
    constructor_track = parent_run/'reference_screen'/candidate_id/'trajectory.csv'
    agent_track = parent_run/'candidates'/candidate_id/'trajectory.csv'
    if constructor_track.is_file() == agent_track.is_file():
        raise ValueError('Require exactly one constructor or agent candidate layout')
    parent_track = constructor_track if constructor_track.is_file() else agent_track
    frames = json.loads((input_dir/'metadata/frames.json').read_text())
    road = load_scene_road(input_dir, frames)
    official_path = input_dir/'metadata/map_drivable_official.json'
    road_official = load_official_drivable(official_path) if official_path.is_file() else None
    from ego_lane_markings import load_no_cross_lines
    no_cross_lines = load_no_cross_lines(input_dir/'metadata/no_cross_lines.json')
    source_path = parent_run/'reference_screen/summary.json'
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    if not source and (parent_run/'config.json').is_file():
        parent_config=json.loads((parent_run/'config.json').read_text())
        source=json.loads((Path(parent_config['constructor_run'])/'reference_screen/summary.json').read_text())
        if parent_config.get('acceptance_policy') != source.get('acceptance_policy'):
            raise ValueError('Editor and constructor policies differ')
    acceptance = source.get('acceptance_policy')
    if acceptance is not None and 'profile' not in acceptance:
        acceptance = None
    allow_safe_stop = bool(source.get('allow_safe_stop', False))
    if acceptance is not None:
        road_margin_target_m = max(road_margin_target_m, acceptance['official_margin_m'])
    if source.get('road_source') == 'official_union':
        if road_official is None:
            raise ValueError('Constructor requires the official drivable map')
        road = RoadGeometry(road_official.union, map_source=str(official_path),
                            polygon_count=len(road_official.polygons))
    times, baseline_xy, baseline_yaw, actors = scene_from_frames(frames, target_track=target_track)
    observation_end_s=(acceptance or {}).get('observation_end_s',3.)
    freeze_before_s=observation_end_s if freeze_before_s is None else float(freeze_before_s)
    if not np.isfinite(freeze_before_s):
        raise ValueError('FOCI freeze boundary must be finite')
    # Freeze-before restricts optimizer control points; the window additionally
    # guarantees exact unchanged sampled history and outside-window parent poses.
    edit_window_s=list(edit_window_s) if edit_window_s is not None else [max(observation_end_s,freeze_before_s),float(times[-1])]
    if max_candidates is not None and (type(max_candidates) is not int or not 1<=max_candidates<=8):
        raise ValueError('FOCI candidate limit must be in [1,8]')
    if type(max_iterations) is not int or not 1<=max_iterations<=250:
        raise ValueError('FOCI max_iterations must be in [1,250]')
    if type(preserve_route) is not bool or terminal_behavior not in ('free','stop') or improvement_goal not in ('balanced','recover_progress','smoother_motion'):
        raise ValueError('Invalid FOCI edit goals')
    edit_goals=dict(preserve_route=preserve_route,terminal_behavior=terminal_behavior,improvement_goal=improvement_goal)
    if speed_targets:
        from speed_profile_contract import validate_targets
        validate_targets(speed_targets,start=edit_window_s[0],end=edit_window_s[1],max_speed=(acceptance or policy.resolve())['max_speed_mps'])
        edit_goals['speed_targets']=speed_targets
    parent_xy, parent_yaw = read_track(parent_track, times)
    (run_dir/'input').mkdir()
    shutil.copyfile(parent_track, run_dir/'input/parent-trajectory.csv')
    road_weights = (300., 1000.) if road_official is not None else (0.,)
    config = dict(acceptance_policy=acceptance, version='ego_foci_probe.v8',edit_goals=edit_goals, input_dir=str(input_dir), parent_run=str(parent_run),
        parent_candidate_id=candidate_id, target_track=target_track,
        method='FOCI-inspired dynamic 2D Gaussian overlap with cubic B-spline control points; not the original 3D FOCI implementation',
        vehicle_model='Three anisotropic 2D Gaussians per vehicle; longitudinal sigma=segment_length/2 and lateral sigma=width/2',
        gaussian_overlap='Normalized 2D Gaussian overlap integral summed over nine ego-target pairs at each observed timestamp',
        road_constraint=('quadratic penalty on each body corner falling below the penalty target margin '
            'inside the official nuPlan drivable union; final acceptance still uses the hard official-map gate'),
        road_margin_target_m=road_margin_target_m, road_gate_margin_m=(acceptance['official_margin_m'] if acceptance else policy.OFFICIAL_MARGIN_M),
        allow_safe_stop=allow_safe_stop, road_source=source.get('road_source', 'scene_lane_union'),
        road_weights=list(road_weights) if road_official is not None else [0.],
        control_count=16, freeze_before_s=freeze_before_s, edit_window_s=edit_window_s,
        observation_end_s=observation_end_s, max_candidates=max_candidates, xy_control_bound_m=2., yaw_control_bound_rad=.35,
        lateral_soft_target_mps=.75, lateral_hard_limit_mps=1.,
        max_iterations=max_iterations, observed_sample_count=len(times), carla_executed=False,
        api_used=False, omnisg_rendered=False)
    from defense_behavior import DrivingObjective,preferences
    behavior=DrivingObjective(times,baseline_xy,actors,target_track,acceptance or policy.resolve())
    parent_ids=list(dict.fromkeys([candidate_id,*(additional_parents or [])]))
    if len(parent_ids)>(max_candidates or 8):
        raise ValueError('FOCI parent count exceeds candidate budget')
    config.update(parent_candidate_ids=parent_ids,behavior_preferences=preferences(),
        terminal_window_policy='terminal free only when editable window ends at scene end; sampled history and any later outside-window parent poses stay fixed')
    atomic_json(run_dir/'config.json', config)
    problem = FociProblem(times, parent_xy, parent_yaw, actors, target_track=target_track,
        official_road=road_official, freeze_before_s=freeze_before_s,
        road_margin_m=road_margin_target_m,edit_window_s=edit_window_s,observation_end_s=observation_end_s,behavior_objective=behavior,edit_goals=edit_goals)
    weight_sets = [(1., .03, 1., 1.), (1., .03, 1., 10.), (1., .03, 1., 100.),
        (3., .03, 1., 1.), (3., .03, 1., 10.), (3., .03, 1., 100.),
        (10., .01, .1, 10.), (30., .003, .1, 100.), (100., .001, .01, 1000.)]
    if road_official is not None:
        base_sets = [(1., .03, 1., 100.), (3., .03, 1., 10.), (10., .01, .1, 100.),
            (100., .001, .01, 1000.)]
        weight_sets = [(*base, rw) for rw in road_weights for base in base_sets]
    else:
        weight_sets = [(*base, 0.) for base in weight_sets]
    if max_candidates is not None:
        weight_sets=weight_sets[:max_candidates]
    candidates = []
    def archive(cid, xy, yaw, optimization):
        folder = run_dir/'candidates'/cid; folder.mkdir(parents=True)
        write_track(folder/'trajectory.csv', times, xy, yaw)
        result = screen(times, baseline_xy, baseline_yaw, actors, xy, yaw, target_track=target_track,
            road=road, road_official=road_official, allow_safe_stop=allow_safe_stop, no_cross_lines=no_cross_lines, acceptance=acceptance)
        result['candidate_id'] = cid
        atomic_json(folder/'screen.json', result)
        atomic_json(folder/'optimization.json', optimization)
        eligible = not result['target_overlap_indices'] and all(value for key, value in result['checks'].items() if key != 'target_clearance')
        behavior_metrics=behavior.evaluate(xy,yaw)
        record = dict(candidate_id=cid, parent_candidate_id=optimization.get('parent_candidate_id',candidate_id),
            behavior_metrics=behavior_metrics,min_target_clearance_m=result['min_target_clearance_m'],
            min_other_clearance_m=result['min_other_clearance_m'], target_overlap_indices=result['target_overlap_indices'],
            status=result['status'], eligible_for_best=eligible, optimization=optimization)
        candidates.append(record)
        return result
    parent_result = archive('parent', parent_xy, parent_yaw, dict(source='immutable_parent'))
    archive('spline_initial', problem.zero_xy, problem.zero_yaw, dict(source='zero_edit_spline_fit',
        initial_mean_overlap_per_frame=problem.overlap_scale))
    if problem.overlap_scale is not None or behavior is not None:
        for index, weights in enumerate(weight_sets, 1):
            active_parent=parent_ids[(index-1)%len(parent_ids)]
            if active_parent!=candidate_id:
                path=(parent_run/'reference_screen'/active_parent/'trajectory.csv') if constructor_track.is_file() else parent_run/'candidates'/active_parent/'trajectory.csv'
                active_xy,active_yaw=read_track(path,times)
                saved_parent=run_dir/'input'/('parent-'+active_parent+'.csv')
                if not saved_parent.exists():shutil.copyfile(path,saved_parent)
                active_problem=FociProblem(times,active_xy,active_yaw,actors,target_track=target_track,
                    official_road=road_official,freeze_before_s=freeze_before_s,road_margin_m=road_margin_target_m,
                    edit_window_s=edit_window_s,observation_end_s=observation_end_s,behavior_objective=behavior,edit_goals=edit_goals)
            else:
                active_problem=problem
            xy, yaw, optimization = active_problem.optimize(*weights, max_iterations=max_iterations)
            optimization.update(parent_candidate_id=active_parent,behavior_preferences=preferences(),
                behavior_metrics=behavior.evaluate(xy,yaw))
            if speed_targets:
                from ego_speed_schedule import sampled_target_errors
                optimization['speed_target_feedback']=sampled_target_errors(times,xy,yaw,speed_targets)
            archive(f'foci_{index:03d}', xy, yaw, optimization)
            print(json.dumps({'candidate_id':f'foci_{index:03d}', 'min_target_clearance_m':candidates[-1]['min_target_clearance_m'],
                'status':candidates[-1]['status'], 'solver_success':optimization['success']}))
    else:
        print(json.dumps({'status':'gaussian_optimizer_skipped',
            'reason':'initial_gaussian_overlap_not_finite_positive'}))
    eligible = [record for record in candidates if record['eligible_for_best']]
    if eligible:
        best = min(eligible, key=lambda record:record['behavior_metrics']['optimization_cost'])
        best_result = json.loads((run_dir/'candidates'/best['candidate_id']/'screen.json').read_text())
        improvement = best['min_target_clearance_m']-parent_result['min_target_clearance_m']
        threshold_met = best_result['status'] == 'reference_screen_passed'
        best_candidate_id = best['candidate_id']
        best_min_target_clearance_m = best['min_target_clearance_m']
    else:
        best_candidate_id = None; best_min_target_clearance_m = None; improvement = None; threshold_met = False
    if threshold_met:
        status = 'reference_probe_completed'
    elif improvement is not None and improvement > 1e-9:
        status = 'reference_probe_completed_below_threshold'
    else:
        status = 'reference_probe_completed_no_improvement'
    summary = dict(status=status,
        best_candidate_id=best_candidate_id, selected_candidate_id=best_candidate_id if threshold_met else None,
        parent_candidate_id=candidate_id, parent_min_target_clearance_m=parent_result['min_target_clearance_m'],
        best_min_target_clearance_m=best_min_target_clearance_m, improvement_over_parent_m=improvement,
        clearance_policy='Shared progress/reserve optimization; excess clearance has no reward. No added acceptance gate.',
        behavior_preferences=preferences(),parent_candidate_ids=parent_ids,
        reference_checks_passed=threshold_met, threshold_met=threshold_met, candidate_count=len(candidates),
        gaussian_optimizer='executed' if problem.overlap_scale is not None or behavior is not None else 'skipped_initial_overlap_not_finite_positive',
        physical_avoidance='undetermined', road_status='checked',
        rear_coverage='partial_observations_only', carla_executed=False, omnisg_rendered=False,
        candidates=candidates,
        scope='Observed 2 Hz actor geometry and whole-body road checks densified to 0.1 s. Gaussian cost guides optimization but does not determine acceptance. Progress checks compare cumulative traveled path length, not net forward displacement. No continuous-time collision, CARLA or physical-safety certification.')
    atomic_json(run_dir/'summary.json', summary)
    return summary



def verify_candidate(run_dir, candidate_id, *, input_dir, target_track, acceptance, times, actors, parent_xy, parent_yaw):
    run_dir=Path(run_dir)
    config=json.loads((run_dir/'config.json').read_text())
    if (config.get('version') not in ('ego_foci_probe.v5','ego_foci_probe.v6','ego_foci_probe.v7','ego_foci_probe.v8') or Path(config['input_dir']).resolve()!=Path(input_dir).resolve()
            or config['target_track']!=target_track or config['acceptance_policy']!=acceptance):
        raise ValueError('FOCI candidate scene, policy or version differs')
    optimization=json.loads((run_dir/'candidates'/candidate_id/'optimization.json').read_text())
    selected_parent=optimization.get('parent_candidate_id',config['parent_candidate_id'])
    if selected_parent not in config.get('parent_candidate_ids',[config['parent_candidate_id']]):
        raise ValueError('FOCI candidate parent not declared in run')
    parent_file='parent-trajectory.csv' if selected_parent==config['parent_candidate_id'] else 'parent-'+selected_parent+'.csv'
    saved_xy,saved_yaw=read_track(run_dir/'input'/parent_file,times)
    behavior=None
    if config['version'] in ('ego_foci_probe.v6','ego_foci_probe.v7','ego_foci_probe.v8'):
        from defense_behavior import DrivingObjective,preferences
        if config.get('behavior_preferences')!=preferences():
            raise ValueError('FOCI behavior preferences differ from archived run')
        _,reference,_,_=scene_from_frames(json.loads((Path(input_dir)/'metadata/frames.json').read_text()),target_track)
        behavior=DrivingObjective(times,reference,actors,target_track,acceptance or policy.resolve())
    if not np.array_equal(saved_xy,parent_xy) or not np.array_equal(saved_yaw,parent_yaw):
        raise ValueError('FOCI candidate parent trajectory differs')
    official_path=Path(input_dir)/'metadata/map_drivable_official.json'
    official=load_official_drivable(official_path) if official_path.is_file() else None
    edit_goals=config.get('edit_goals',{})
    problem=FociProblem(times,parent_xy,parent_yaw,actors,control_count=config['control_count'],
        freeze_before_s=config['freeze_before_s'],target_track=target_track,official_road=official,
        road_margin_m=config['road_margin_target_m'],edit_window_s=config['edit_window_s'],
        observation_end_s=config['observation_end_s'],behavior_objective=behavior,edit_goals=edit_goals)
    optimization=json.loads((run_dir/'candidates'/candidate_id/'optimization.json').read_text())
    delta=np.asarray(optimization['control_delta'],float)
    if (delta.shape!=problem.initial_control.shape or not np.isfinite(delta).all()
            or np.any(np.abs(delta[problem.frozen])>1e-12)):
        raise ValueError('Invalid FOCI optimizer controls')
    bounds=np.asarray(problem.bounds()).reshape(delta.shape+(2,))
    if np.any(delta<bounds[:,:,0]-1e-12) or np.any(delta>bounds[:,:,1]+1e-12):
        raise ValueError('FOCI controls exceed their archived bounds')
    expected_xy,expected_yaw=problem.trajectory(delta)
    xy,yaw=read_track(run_dir/'candidates'/candidate_id/'trajectory.csv',times)
    if not np.allclose(xy,expected_xy,atol=1e-9,rtol=0) or not np.allclose(yaw,expected_yaw,atol=1e-9,rtol=0):
        raise ValueError('FOCI saved trajectory differs from archived controls')
    return xy,yaw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--max-iterations', type=int, default=250)
    parser.add_argument('--target-track', required=True)
    parser.add_argument('--freeze-before', type=float, default=None)
    parser.add_argument('--edit-window', type=float, nargs=2)
    parser.add_argument('--max-candidates', type=int)
    parser.add_argument('--speed-targets',type=Path,help='JSON timed signed speed states; soft FOCI preferences, not acceptance gates')
    parser.add_argument('--preserve-route',action='store_true')
    parser.add_argument('--terminal-behavior',choices=['free','stop'],default='free')
    parser.add_argument('--improvement-goal',choices=['balanced','recover_progress','smoother_motion'],default='balanced')
    parser.add_argument('--additional-parents',nargs='*',default=[])
    parser.add_argument('--road-margin-target', type=float, default=0.15,
        help='penalty target margin inside the official drivable union; the hard gate uses 0.1 m')
    args = parser.parse_args()
    summary = run(args.input, args.parent_run, args.candidate, args.run, args.max_iterations,
        args.target_track, args.freeze_before, args.road_margin_target,args.edit_window,args.max_candidates,args.additional_parents,args.preserve_route,args.terminal_behavior,args.improvement_goal,
        json.loads(args.speed_targets.read_text()) if args.speed_targets else None)
    print(json.dumps({key:summary[key] for key in ('status', 'best_candidate_id', 'best_min_target_clearance_m',
        'improvement_over_parent_m', 'threshold_met')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
