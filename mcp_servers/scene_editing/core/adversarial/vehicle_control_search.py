"""Parent-preserving bicycle controls and constraint-guided numerical search."""
import numpy as np
import shapely
from scipy.optimize import least_squares

from attack_objective import (best_hold, frame_errors, relative_state)
from attack_task_contract import corridor_errors
from run_ego_reference import EGO_FRONT, EGO_REAR, EGO_WIDTH
from vehicle_initial_state import initial_state, initial_lateral_velocity
from vehicle_trajectory import actor_observation_series, actor_pose_at_time, sample_xy


def wrap(values):
    return np.arctan2(np.sin(values), np.cos(values))


def infer_controls(times, xy, yaw, size, boundary):
    wheelbase = .6 * float(size[0])
    rear = xy - wheelbase/2 * np.column_stack([np.cos(yaw), np.sin(yaw)])
    velocity = np.gradient(rear, times, axis=0)
    speed = np.maximum(0., velocity[:, 0]*np.cos(yaw) + velocity[:, 1]*np.sin(yaw))
    steering = np.arctan(wheelbase*np.gradient(np.unwrap(yaw), times)/np.maximum(speed, .25))
    steering[speed < .1] = 0.
    query = np.r_[np.arange(boundary, times[-1]-.001, .75), times[-1]]
    return dict(times_s=query.tolist(), acceleration_mps2=np.interp(query, times, np.gradient(speed, times)).clip(-4.5, 2.5).tolist(),
                steering_rad=np.interp(query, times, steering).clip(-.55, .55).tolist())


def integrate_controls(times, seed_xy, seed_yaw, dense_times, size, controls, boundary=3., acceptance=None):
    import acceptance_policy as policy
    effective = policy.resolve(acceptance)
    speed_limit = effective['generator_max_speed_mps']
    query = np.unique(np.r_[times, dense_times, boundary])
    center = sample_xy(times, seed_xy, [boundary])[0]
    initial = initial_state(times, seed_xy, seed_yaw, size, boundary, effective)
    angle, speed = initial['yaw'], initial['speed']
    if speed > speed_limit+1e-6:
        raise ValueError('unsupported_initial_state: speed exceeds configured generator range')
    wheelbase = .6*float(size[0])
    rear = initial['rear_xy'].copy()
    acceleration, steering = 0., initial['steering']
    positions, angles, actual = [], [], []
    previous = boundary
    ct = np.asarray(controls['times_s'])
    ca, cs = np.asarray(controls['acceleration_mps2']), np.asarray(controls['steering_rad'])
    if ct.ndim != 1 or len(ct)<2 or ca.shape!=ct.shape or cs.shape!=ct.shape or not np.isfinite(np.r_[ct,ca,cs]).all() or np.any(np.diff(ct)<=0) or ct[0]!=boundary or abs(ct[-1]-times[-1])>1e-6:
        raise ValueError('Controls need finite ordered arrays covering observation boundary to scene end')
    if np.any(abs(cs)>.55+1e-9) or np.any(ca < -4.5-1e-9) or np.any(ca > 2.5+1e-9):
        raise ValueError('Control commands exceed acceleration or steering bounds')
    for time in query:
        if time <= boundary:
            positions.append(sample_xy(times, seed_xy, [time])[0])
            angles.append(float(np.interp(time, times, np.unwrap(seed_yaw))))
            actual.append([speed, 0., 0.])
            continue
        steps = max(1, int(np.ceil((time-previous)/.025)))
        dt = (time-previous)/steps
        for index in range(steps):
            now = previous+(index+.5)*dt
            control_index = min(len(ct)-1, int(np.searchsorted(ct, now, side='right')-1))
            acceleration += max(-4.*dt, min(4.*dt, ca[control_index]-acceleration))
            limit = min(.55, float(np.arctan(3.4*wheelbase/max(speed**2, .1))))
            desired = max(-limit, min(limit, cs[control_index]))
            steering += max(-.7*dt, min(.7*dt, desired-steering))
            following = max(0., min(speed_limit, speed+acceleration*dt))
            middle_speed = (speed+following)/2
            yaw_rate = middle_speed/wheelbase*np.tan(steering)
            middle_angle = angle+yaw_rate*dt/2
            rear += dt * (middle_speed * np.array([np.cos(middle_angle), np.sin(middle_angle)])
                          + initial_lateral_velocity(initial, now) * np.array([-np.sin(middle_angle), np.cos(middle_angle)]))
            angle += yaw_rate*dt
            speed = following
        positions.append(rear+wheelbase/2*np.array([np.cos(angle), np.sin(angle)]))
        angles.append(angle)
        actual.append([speed, acceleration, steering])
        previous = float(time)
    positions, angles, actual = np.asarray(positions), np.asarray(angles), np.asarray(actual)
    native = (sample_xy(query, positions, times), np.interp(times, query, angles))
    dense = (sample_xy(query, positions, dense_times), np.interp(dense_times, query, angles))
    state = np.column_stack([np.interp(dense_times, query, actual[:, axis]) for axis in range(3)])
    return (*native, *dense), state


def preservation_evidence(times, xy, yaw, parent_xy, parent_yaw, arguments):
    window = arguments.get('edit_window_s', [3., float(times[-1])])
    outside = (times < window[0]-1e-8) | (times > window[1]+1e-8)
    position = float(np.linalg.norm(xy[outside]-parent_xy[outside], axis=1).max()) if outside.any() else 0.
    angle = float(abs(wrap(yaw[outside]-parent_yaw[outside])).max()) if outside.any() else 0.
    velocity = np.gradient(xy, times, axis=0)
    parent_velocity = np.gradient(parent_xy, times, axis=0)
    speed = float(np.linalg.norm(velocity[outside]-parent_velocity[outside], axis=1).max()) if outside.any() else 0.
    passed = not arguments.get('preserve_outside_window', False) or (position <= .05 and angle <= .05 and speed <= .1)
    return dict(passed=bool(passed), requested=arguments.get('preserve_outside_window', False),
                max_outside_position_error_m=position, max_outside_yaw_error_rad=angle,
                max_outside_velocity_error_mps=speed,
                tolerances=dict(position_m=.05, yaw_rad=.05, velocity_mps=.1),
                scope='Entire integrated trajectory is retained; no discontinuous endpoint splice')


def signed_box_gap(xy, yaw, size, other_xy, other_yaw, other_size):
    delta = other_xy-xy
    axes = np.stack([np.column_stack([np.cos(yaw), np.sin(yaw)]),
                     np.column_stack([-np.sin(yaw), np.cos(yaw)]),
                     np.column_stack([np.cos(other_yaw), np.sin(other_yaw)]),
                     np.column_stack([-np.sin(other_yaw), np.cos(other_yaw)])], axis=1)
    own_x = np.column_stack([np.cos(yaw), np.sin(yaw)])
    own_y = np.column_stack([-np.sin(yaw), np.cos(yaw)])
    other_x = np.column_stack([np.cos(other_yaw), np.sin(other_yaw)])
    other_y = np.column_stack([-np.sin(other_yaw), np.cos(other_yaw)])
    projection = np.abs(np.einsum('nij,nj->ni', axes, delta))
    own = np.abs(np.einsum('nij,nj->ni', axes, own_x))*size[0]/2 + np.abs(np.einsum('nij,nj->ni', axes, own_y))*size[1]/2
    other = np.abs(np.einsum('nij,nj->ni', axes, other_x))*np.asarray(other_size)[..., 0, None]/2 + np.abs(np.einsum('nij,nj->ni', axes, other_y))*np.asarray(other_size)[..., 1, None]/2
    return np.max(projection-own-other, axis=1)


class ControlSearch:
    """Cheap continuous guidance; final acceptance always uses full checkers."""
    def __init__(self, editor, arguments):
        self.editor, self.arguments = editor, arguments
        self.parent = editor.nodes[arguments['parent_candidate_id']]
        self.times = editor.dense_times
        self.ego = sample_xy(editor.times, editor.ego, self.times)
        self.ego_yaw = np.interp(self.times, editor.times, editor.ego_yaw)
        self.ego_center = self.ego+(EGO_FRONT-EGO_REAR)/2*np.column_stack([np.cos(self.ego_yaw), np.sin(self.ego_yaw)])
        self.boundary = editor.constraints.get('observation_end_s', 3.)
        self.base = self.parent['result'].get('control_sequence') or infer_controls(self.times,
            self.parent['dense_xy'], self.parent['dense_yaw'], editor.target['size_m'], self.boundary)
        self.control_times = np.asarray(self.base['times_s'])
        window = arguments.get('edit_window_s', [self.boundary, float(self.times[-1])])
        self.editable = np.flatnonzero((self.control_times >= window[0]) & (self.control_times < window[1]))
        if not len(self.editable):
            raise ValueError('edit_window_s contains no control knots; widen it')
        self.obstacles = []
        for _, observed, poses in actor_observation_series(editor.times, editor.actors, editor.target['track_id']):
            samples = [actor_pose_at_time(observed, poses, time, 1.1*np.median(np.diff(editor.times)), scene_times=editor.times) for time in self.times]
            mask = np.array([pose is not None for pose in samples])
            if mask.any():
                self.obstacles.append((mask, np.asarray([pose for pose in samples if pose is not None])))
        self.n = len(self.editable)
        self.initial = np.r_[np.asarray(self.base['acceleration_mps2'])[self.editable], np.asarray(self.base['steering_rad'])[self.editable]*10]
        self.lower = np.r_[np.full(self.n, -4.5), np.full(self.n, -5.5)]
        self.upper = np.r_[np.full(self.n, 2.5), np.full(self.n, 5.5)]

    def controls(self, values):
        acceleration = np.asarray(self.base['acceleration_mps2']).copy()
        steering = np.asarray(self.base['steering_rad']).copy()
        acceleration[self.editable] = values[:self.n]
        steering[self.editable] = values[self.n:]/10
        return dict(times_s=self.control_times.tolist(), acceleration_mps2=acceleration.tolist(), steering_rad=steering.tolist())

    def simulate(self, controls):
        return integrate_controls(self.editor.times, self.editor.seed_xy, self.editor.seed_yaw,
            self.times, self.editor.target['size_m'], controls, self.boundary, acceptance=self.editor.constraints.get('acceptance_policy'))

    def residual(self, arrays, state):
        xy, yaw = arrays[2:]
        relative = relative_state(self.times, xy, yaw, self.ego, self.ego_yaw)
        residuals = []
        contract = self.editor.constraints.get('task_contract')
        objectives = [(self.arguments['objective'], .5 if contract else 3.)]
        if contract:
            objectives.append((contract['objective'], 6.))
        if contract and contract.get('path_corridor'):
            errors, _, _ = corridor_errors(xy, yaw, self.ego, self.ego_yaw, contract['path_corridor'])
            mask = self.times >= contract['path_corridor']['start_s']-.001
            residuals.extend((8*errors[mask]).ravel())
        for current, weight in objectives:
            for frame in current['keyframes']:
                errors = frame_errors(frame, relative)
                _, start, end = best_hold(self.times, frame['time_window_s'], frame['hold_duration_s'], errors)
                # Fixed residual length across changing best hold locations.
                residuals.extend(weight*np.sqrt(np.mean(errors[start:end+1]**2, axis=0)+1e-12))
        objective = contract['objective'] if contract else self.arguments['objective']
        gap = signed_box_gap(xy, yaw, self.editor.target['size_m'], self.ego_center, self.ego_yaw, [EGO_FRONT+EGO_REAR, EGO_WIDTH])
        window = objective['interaction_window_s']
        mask = (self.times >= window[0]) & (self.times <= window[1])
        if objective['interaction'] == 'allowed':
            forbidden = self.times <= self.editor.constraints.get('observation_end_s', 3.)
        elif objective['interaction'] == 'either':
            residuals.extend([5*max(0., float(gap[mask].min())-1.5)])
            forbidden = self.times < window[0]
        elif objective['interaction'] == 'contact':
            residuals.extend([5*max(0., float(gap[mask].min())+.03)])
            forbidden = self.times < window[0]
        else:
            residuals.extend([3*max(0., float(gap[mask].min())-1.5)])
            forbidden = np.ones(len(self.times), dtype=bool)
        residuals.extend(5*np.maximum(.17-gap[forbidden], 0.))
        # Signed corner/edge-midpoint distance guides away from official map edges.
        shape = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 0], [-1, 0], [0, 1], [0, -1]])*np.asarray(self.editor.target['size_m'][:2])/2
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        corners = xy[:, None, :]+np.stack([c*shape[:, 0]-s*shape[:, 1], s*shape[:, 0]+c*shape[:, 1]], axis=-1)
        points = shapely.points(corners.reshape(-1, 2))
        road = self.editor.road.union
        distance = shapely.distance(points, road.boundary)
        signed = np.where(shapely.covers(road, points), distance, -distance)
        residuals.extend(5*np.maximum(self.editor.constraints.get('official_margin_m', .1)+.03-signed, 0.)/np.sqrt(8))
        for observed, poses in self.obstacles:
            separation = signed_box_gap(xy[observed], yaw[observed], self.editor.target['size_m'], poses[:, :2], poses[:, 2], poses[:, 3:5])
            residuals.extend(5*np.maximum(.12-separation, 0.))
        if self.arguments.get('preserve_outside_window'):
            lower, upper = self.arguments['edit_window_s']
            outside = (self.times < lower) | (self.times > upper)
            residuals.extend((20*(xy[outside]-self.parent['dense_xy'][outside])).ravel())
            residuals.extend(20*wrap(yaw[outside]-self.parent['dense_yaw'][outside]))
            velocity_error = np.gradient(xy, self.times, axis=0)-np.gradient(self.parent['dense_xy'], self.times, axis=0)
            residuals.extend((10*velocity_error[outside]).ravel())
        # Small tie-breakers prefer preserving parent and smooth control changes.
        residuals.extend((.02*(xy-self.parent['dense_xy'])).ravel())
        residuals.extend(.05*np.diff(state[:, 1]))
        residuals.extend(.1*np.diff(state[:, 2]))
        return np.asarray(residuals)

    def starts(self):
        yield self.initial.copy()
        # Multiple general acceleration/turning guesses; no maneuver-specific path.
        for braking in (-1., -3., .5):
            values = self.initial.copy()
            values[:self.n] = braking
            yield values.clip(self.lower, self.upper)
        for turn in (-2., 2.):
            values = self.initial.copy()
            values[self.n:] += turn
            yield values.clip(self.lower, self.upper)

    def optimize(self, evaluate, limit, reserve=None):
        """evaluate debits and archives each actual objective evaluation."""
        class BudgetReached(Exception):
            pass
        records = []

        def fun(values):
            if len(records) >= limit:
                raise BudgetReached
            controls = self.controls(values)
            if reserve is not None:
                reserve(controls)
            arrays, state = self.simulate(controls)
            residual = self.residual(arrays, state)
            record = evaluate(controls, arrays, state, float(residual @ residual))
            records.append((float(residual @ residual), values.copy(), record))
            return residual

        starts = list(self.starts())
        for start in starts[:limit]:
            fun(start)
        best = sorted(records, key=lambda record: record[0])[:2]
        for index, (_, start, _) in enumerate(best):
            local_limit = len(records)+(limit-len(records))//(2-index)
            def local_fun(values):
                if len(records) >= local_limit:
                    raise BudgetReached
                return fun(values)
            try:
                least_squares(local_fun, start, bounds=(self.lower, self.upper),
                    diff_step=.01, ftol=1e-4, xtol=1e-4, gtol=1e-4, max_nfev=20)
            except BudgetReached:
                pass
        return sorted(records, key=lambda record: record[0])
