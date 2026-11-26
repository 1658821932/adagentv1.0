"""Parent-preserving bicycle controls and constraint-guided numerical search."""
import numpy as np
import acceptance_policy as policy
import shapely
from scipy.optimize import least_squares

from attack_objective import (best_hold, frame_errors, relative_state)
from attack_task_contract import corridor_errors
from trajectory_waypoints import measure_waypoints, time_bounds
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


def _control_initial_state(times, xy, yaw, size, boundary, acceptance):
    state = initial_state(times, xy, yaw, size, boundary, acceptance)
    if state['speed'] > acceptance['generator_max_speed_mps'] + 1e-6:
        raise ValueError('unsupported_initial_state: speed exceeds configured generator range')
    return state


def integrate_controls(times, seed_xy, seed_yaw, dense_times, size, controls, boundary=3., acceptance=None):
    import acceptance_policy as policy
    effective = policy.resolve(acceptance)
    speed_limit = effective['generator_max_speed_mps']
    query = np.unique(np.r_[times, dense_times, boundary])
    center = sample_xy(times, seed_xy, [boundary])[0]
    initial = _control_initial_state(times, seed_xy, seed_yaw, size, boundary, effective)
    angle, speed = initial['yaw'], initial['speed']
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


def event_window_guidance(times, gap, window, required=False):
    lower,upper=map(float,window)
    mask=(times>=lower-.001)&(times<=upper+.001)
    if not mask.any():
        raise ValueError('Event window contains no dense samples')
    contacts=np.flatnonzero(gap<0.)
    first_contact=None if not len(contacts) else float(times[contacts[0]])
    minimum=float(gap[mask].min())
    penetration=max(0.,-minimum)
    late=0. if first_contact is None else max(0.,first_contact-upper)
    proximity_weight=10. if required else 2.5
    late_weight=8. if required else 4.
    penetration_res = (4.*max(0.,.02-penetration) + 3.*max(0.,penetration-0.5)) if required else 0.
    residuals=[proximity_weight*max(0.,minimum),late_weight*late,penetration_res]
    metrics=dict(window_min_signed_gap_m=minimum,window_penetration_m=penetration,
        window_contact_sample_count=int(np.count_nonzero(gap[mask]<0.)),
        window_reference_contact=bool(np.any(gap[mask]<0.)),
        first_reference_contact_time_s=first_contact,late_contact_seconds=late)
    return residuals,metrics


def release_starts(control_times, editable, initial, target_time, other_time=None):
    """Continuous brake/release guesses; times come from the current rollout.

    These are controls before physical execution, not a collision-response model.
    No poses are spliced and the complete trajectory still needs strict checks.
    """
    knots=np.asarray(control_times)[editable]
    spacing=float(np.median(np.diff(control_times)))
    # Releasing braking before contact can be necessary: the actuator needs time
    # to ramp from negative to positive acceleration even while speed is zero.
    releases=[target_time-offset*spacing for offset in (4,3,2,1,0,-1)]
    if other_time is not None:
        releases.append((target_time+other_time)/2)
    seen=set()
    n=len(editable)
    for release in releases:
        after=knots>=release-1e-6
        if not after.any() or after.all():continue
        for acceleration in (2.5,1.5):
            values=initial.copy()
            values[:n][after]=acceleration
            key=tuple(values)
            if key in seen:continue
            seen.add(key)
            yield values
    # Small temporary steering alternatives also return to the parent controls.
    # They are proposals, never permission to leave the drivable surface.
    release=target_time-spacing
    after=knots>=release-1e-6
    pulse=after & (knots<target_time+spacing)
    for steering in (-.03,.03):
        if not pulse.any() or after.all():continue
        values=initial.copy()
        values[:n][after]=2.5
        values[n:][pulse]+=10*steering
        key=tuple(values)
        if key not in seen:
            seen.add(key)
            yield values


def refinement_groups(values,lower,upper,n):
    """At most eight shared control offsets, with exact command bounds.

    A full per-knot finite-difference Jacobian can exhaust a small branch before
    taking any optimization step. Preserve its seed shape and vary segments.
    """
    cuts=np.flatnonzero(np.abs(np.diff(values[:n]))>.05)+1
    acceleration=np.split(np.arange(n),cuts)
    if len(acceleration)>4:acceleration=np.array_split(np.arange(n),4)
    candidates=acceleration+[group+n for group in np.array_split(np.arange(n),min(4,n))]
    groups=[];bounds=[]
    for group in candidates:
        if not len(group):continue
        lo=float(np.max(lower[group]-values[group]));hi=float(np.min(upper[group]-values[group]))
        if hi-lo<=1e-8:continue
        groups.append(group);bounds.append((lo,hi))
    return groups,np.asarray(bounds)


def motion_guidance(editor,times,xy,yaw,initial_seed=None):
    from trajectory_checks import screen_reference
    from vehicle_motion import check_bicycle
    from vehicle_trajectory import motion_samples
    effective=policy.resolve(editor.constraints.get('acceptance_policy'))
    seed_xy,seed_yaw=(editor.seed_xy,editor.seed_yaw) if initial_seed is None else initial_seed
    values=motion_samples(editor.times,times,xy,editor.edit_start_s)
    initial_velocity=(seed_xy[1]-seed_xy[0])/(editor.times[1]-editor.times[0])
    limits=dict(max_speed_mps=effective['max_speed_mps'],
        max_acceleration_mps2=effective['max_acceleration_mps2'],
        max_lateral_acceleration_mps2=effective['attack_max_lateral_acceleration_mps2'],
        max_yaw_rate_radps=policy.ATTACK_MAX_YAW_RATE_RADPS,
        max_initial_velocity_error_mps=policy.ATTACK_MAX_INITIAL_SPEED_ERROR_MPS)
    reports=(screen_reference(values,xy[-1],initial_velocity,limits),
        check_bicycle(times,xy,yaw,editor.target['size_m']))
    residuals=[]
    violations=0
    names=(('max_speed_mps','max_acceleration_mps2','max_lateral_acceleration_mps2',
            'max_yaw_rate_radps','max_initial_velocity_error_mps'),
           ('no_reverse','no_lateral_slide','steering_angle','steering_rate','no_stationary_rotation'))
    for report,keys in zip(reports,names):
        measured={item.get('metric',item.get('check')):item for item in report['violations']}
        for key in keys:
            item=measured.get(key)
            if item is None:
                residuals.append(0.)
                continue
            limit=max(abs(float(item['limit'])),1e-6)
            excess=(float(item['actual'])-float(item['limit']))/limit
            residuals.append(20.*max(0.,excess))
            violations+=1
    return residuals,dict(motion_guidance_violation_count=violations)


_CURRENT_POSE = object()
_FORK_SEARCH_CTX = None


def _fork_eval_one(item):
    spec, values = item
    search = _FORK_SEARCH_CTX
    search.activate(spec)
    controls = search.controls(values)
    arrays, state = search.simulate(controls)
    residual = search.residual(arrays, state)
    return (spec, values, controls, arrays, state, residual, dict(search.last_metrics))


class ControlSearch:
    """Cheap continuous guidance; final acceptance always uses full checkers."""
    def __init__(self, editor, arguments):
        self.editor, self.arguments = editor, arguments
        self.parent = editor.nodes[arguments['parent_candidate_id']]
        from initial_pose_edit import parent_seed, search_specs
        self.active_pose=None
        self.active_seed_xy,self.active_seed_yaw,_=parent_seed(editor,self.parent)
        self.times = editor.dense_times
        self.ego = sample_xy(editor.times, editor.ego, self.times)
        self.ego_yaw = np.interp(self.times, editor.times, editor.ego_yaw)
        self.ego_center = self.ego+(EGO_FRONT-EGO_REAR)/2*np.column_stack([np.cos(self.ego_yaw), np.sin(self.ego_yaw)])
        self.boundary = getattr(editor,'edit_start_s',policy.edit_start(editor.constraints.get('acceptance_policy') or editor.constraints))
        self.waypoints = arguments['objective'].get('waypoints', [])
        if self.waypoints and 'initial_pose_search' not in arguments:
            specs = list(self._waypoint_pose_specs())
        else:
            specs = list(search_specs(arguments, editor, self.parent))
        if 'initial_pose_search' not in arguments:
            hint_specs = self._hint_pose_specs()
            if len(hint_specs) > 1:
                seen_xy = [np.asarray(self.active_seed_xy[0], dtype=float)]
                for sp in specs:
                    if sp is None:
                        continue
                    off = (sp or {}).get('position_offset_m') or {}
                    seen_xy.append(self.active_seed_xy[0] + np.array([float(off.get('forward_m', 0.0)), float(off.get('left_m', 0.0))]))
                for sp in hint_specs[1:]:
                    off = (sp or {}).get('position_offset_m') or {}
                    pos = self.active_seed_xy[0] + np.array([float(off.get('forward_m', 0.0)), float(off.get('left_m', 0.0))])
                    if not any(float(np.linalg.norm(pos - prev)) < 2.5 for prev in seen_xy):
                        seen_xy.append(pos)
                        specs.append(sp)
        self.pose_specs = self._eligible_pose_specs(specs)
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

    def _hint_pose_specs(self):
        specs = [None]
        if (self.editor.target.get('operation') != 'add_vehicle'
                or self.boundary != self.editor.times[0]
                or self.arguments.get('preserve_outside_window')):
            return specs
        hint = self.editor.constraints.get('spatial_hint') or {}
        cands = hint.get('attack_start_candidates_xy_m') or []
        if not isinstance(cands, list) or len(cands) < 2:
            return specs
        from connector_route import lane_centerline_seed
        ref_yaw = float(self.active_seed_yaw[0])
        ref_speed = max(3.0, float(self.editor.target.get('initial_speed_mps') or 6.0))
        chosen_xy = [np.asarray(self.active_seed_xy[0], dtype=float)]
        for pt in cands[:3]:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                continue
            hx, hy = float(pt[0]), float(pt[1])
            goal_xy = [float(self.editor.ego[len(self.editor.ego)//2, 0]), float(self.editor.ego[len(self.editor.ego)//2, 1])] if len(self.editor.ego) else None
            snapped = lane_centerline_seed(self.editor.scene, self.editor.frames, hx, hy, ref_yaw, ref_speed, self.editor.times, max_snap_m=10.0, goal_xy=goal_xy)
            if snapped is not None:
                sx, sy = float(snapped[2][0]), float(snapped[2][1])
                syaw = float(snapped[3])
            else:
                sx, sy, syaw = hx, hy, ref_yaw
            pos = np.array([sx, sy], dtype=float)
            if any(float(np.linalg.norm(pos - prev)) < 2.5 for prev in chosen_xy):
                continue
            chosen_xy.append(pos)
            offset = pos - self.active_seed_xy[0]
            specs.append(dict(
                position_offset_m=dict(forward_m=float(offset[0]), left_m=float(offset[1])),
                yaw_mode='explicit',
                yaw_rad=syaw,
            ))
        return specs

    def _waypoint_pose_specs(self):
        first = self.waypoints[0]
        if (self.editor.target.get('operation') != 'add_vehicle'
                or time_bounds(first) != (self.boundary, self.boundary)
                or self.boundary != self.editor.times[0]
                or self.arguments.get('preserve_outside_window')):
            return [None]
        offset = np.asarray(first['position_m']) - self.active_seed_xy[0]
        spec = dict(position_offset_m=dict(forward_m=float(offset[0]), left_m=float(offset[1])))
        if 'speed_mps' in first:
            spec['initial_speed_mps'] = first['speed_mps']
        if 'yaw_rad' in first:
            spec.update(yaw_mode='explicit', yaw_rad=first['yaw_rad'])
        return [None, spec]

    def _eligible_pose_specs(self, specs):
        from initial_pose_edit import adjusted_seed

        editor = self.editor
        effective = policy.resolve(editor.constraints.get('acceptance_policy'))
        self.rejected_initial_poses = []
        eligible = []
        for index, spec in enumerate(specs, 1):
            try:
                xy, yaw, _ = adjusted_seed(editor, self.parent, spec)
                _control_initial_state(editor.times, xy, yaw, editor.target['size_m'],
                    self.boundary, effective)
            except ValueError as error:
                self.rejected_initial_poses.append(dict(variant=index, reason=str(error)))
            else:
                eligible.append(spec)
        if not eligible:
            raise ValueError(self.rejected_initial_poses[0]['reason'])
        return eligible

    def controls(self, values):
        acceleration = np.asarray(self.base['acceleration_mps2']).copy()
        steering = np.asarray(self.base['steering_rad']).copy()
        acceleration[self.editable] = values[:self.n]
        steering[self.editable] = values[self.n:]/10
        return dict(times_s=self.control_times.tolist(), acceleration_mps2=acceleration.tolist(), steering_rad=steering.tolist())

    def activate(self,spec):
        from initial_pose_edit import adjusted_seed
        self.active_pose=spec
        self.active_seed_xy,self.active_seed_yaw,_=adjusted_seed(self.editor,self.parent,spec)

    def simulate(self, controls, initial_pose=_CURRENT_POSE):
        if initial_pose is not _CURRENT_POSE:self.activate(initial_pose)
        return integrate_controls(self.editor.times, self.active_seed_xy, self.active_seed_yaw,
            self.times, self.editor.target['size_m'], controls, self.boundary, acceptance=self.editor.constraints.get('acceptance_policy'))

    def route_start(self,values, first_edit_time=None, *, rear_route=None):
        """Follow the recorded route by geometric lookahead, not wall time.

        This cheap seed estimate obeys the same acceleration ramp. It is not a
        substitute for integrate_controls or full motion/road acceptance.
        """
        wheelbase=.6*float(self.editor.target['size_m'][0])
        route = rear_route
        if route is None:
            route=self.editor.seed_xy-wheelbase/2*np.column_stack([
                np.cos(self.editor.seed_yaw),np.sin(self.editor.seed_yaw)])
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
        if distance[-1]<.5 or np.any(np.diff(distance)<1e-6):return values.copy()
        initial=initial_state(self.editor.times,self.active_seed_xy,self.active_seed_yaw,
            self.editor.target['size_m'],self.boundary,policy.resolve(self.editor.constraints.get('acceptance_policy')))
        rear=initial['rear_xy'].copy();angle=initial['yaw'];steering=initial['steering']
        speed=initial['speed'];acceleration=0.;previous=self.boundary
        commands=self.controls(values)
        angles=np.asarray(commands['steering_rad']).copy()
        maximum=policy.resolve(self.editor.constraints.get('acceptance_policy'))['generator_max_speed_mps']
        segments=np.diff(route,axis=0)
        lengths=np.diff(distance)
        editable = set(self.editable)
        for knot,time in enumerate(self.control_times):
            if time<self.boundary:continue
            steps=max(1,int(np.ceil((time-previous)/.025)))
            dt=(time-previous)/steps
            for step in range(steps):
                now=previous+(step+.5)*dt
                command_index=max(0,int(np.searchsorted(self.control_times,now,side='right')-1))
                acceleration+=np.clip(commands['acceleration_mps2'][command_index]-acceleration,-4.*dt,4.*dt)
                following=np.clip(speed+acceleration*dt,0.,maximum)
                limit=min(.55,float(np.arctan(3.4*wheelbase/max(speed**2,.1))))
                steering+=np.clip(np.clip(angles[command_index],-limit,limit)-steering,-.7*dt,.7*dt)
                middle=(speed+following)/2
                change=middle/wheelbase*np.tan(steering)*dt
                heading=angle+change/2
                rear+=dt*(middle*np.array([np.cos(heading),np.sin(heading)])+
                    initial_lateral_velocity(initial,now)*np.array([-np.sin(heading),np.cos(heading)]))
                angle+=change
                speed=following
            projection=np.sum((rear-route[:-1])*segments,axis=1)/lengths**2
            fractions=np.clip(projection,0.,1.)
            closest=route[:-1]+fractions[:,None]*segments
            index=int(np.argmin(np.linalg.norm(closest-rear,axis=1)))
            along=float(distance[index]+fractions[index]*lengths[index])
            if index==0 and projection[0]<0.:along=float(projection[0]*lengths[0])
            lookahead=max(4.,speed*.8)
            query=along+lookahead
            target=np.array([np.interp(query,distance,route[:,axis]) for axis in range(2)])
            if query<0.:target=route[0]+query*segments[0]/lengths[0]
            if query>distance[-1]:target=route[-1]+(query-distance[-1])*segments[-1]/lengths[-1]
            delta=target-rear
            lateral=-np.sin(angle)*delta[0]+np.cos(angle)*delta[1]
            if knot in editable and (first_edit_time is None or time >= first_edit_time):
                angles[knot]=np.arctan2(2*wheelbase*lateral,max(float(delta@delta),.1))
            previous=float(time)
        result=values.copy()
        result[self.n:]=10*angles[self.editable]
        return result.clip(self.lower,self.upper)

    def waypoint_start(self):
        """Seed the existing steering search from the requested points and their times."""
        points = np.asarray([row['position_m'] for row in self.waypoints])
        times = np.array([sum(time_bounds(row))/2 for row in self.waypoints])
        segments = np.diff(points, axis=0)
        speeds = np.linalg.norm(segments, axis=1) / np.diff(times)
        speeds = np.r_[speeds, speeds[-1]]
        for i, row in enumerate(self.waypoints):
            speeds[i] = row.get('speed_mps', speeds[i])
        rates = np.diff(speeds) / np.diff(times)
        guess = self.initial.copy()
        query = self.control_times[self.editable]
        guess[:self.n] = np.interp(query, (times[:-1]+times[1:])/2, rates, left=rates[0], right=0.)
        # The lookahead seed follows rear-axle positions. Translate the body-center
        # points using the requested heading, or the tangent of the proposed path.
        directions = np.vstack((segments, segments[-1]))
        yaw = np.arctan2(directions[:, 1], directions[:, 0])
        for i, row in enumerate(self.waypoints):
            yaw[i] = row.get('yaw_rad', yaw[i])
        rear = points - .3*self.editor.target['size_m'][0]*np.column_stack([np.cos(yaw), np.sin(yaw)])
        return self.route_start(guess.clip(self.lower, self.upper), rear_route=rear)

    def residual(self, arrays, state):
        self.last_metrics={}
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
            if current.get('waypoints'):
                _, errors = measure_waypoints(self.times, xy, yaw, relative[3], current['waypoints'])
                residuals.extend(max(6., weight)*errors)
        objective = contract['objective'] if contract else self.arguments['objective']
        waypoint_route = bool(self.arguments['objective'].get('waypoints'))
        gap = signed_box_gap(xy, yaw, self.editor.target['size_m'], self.ego_center, self.ego_yaw, [EGO_FRONT+EGO_REAR, EGO_WIDTH])
        window=self.editor.constraints.get('event_window_s',objective['interaction_window_s'])
        event_indices=np.flatnonzero((self.times>=window[0]-.001)&(self.times<=window[1]+.001))
        self.release_target_time=float(self.times[event_indices[np.argmin(gap[event_indices])]])
        contact_indices=event_indices[gap[event_indices]<0.]
        if len(contact_indices):self.release_target_time=float(self.times[contact_indices[0]])
        if getattr(self.editor, 'constraints', {}).get('agent_decides_interaction'):
            # Reference contact guides search only; acceptance remains measurement- and VAD-based.
            window=self.editor.constraints.get('event_window_s',objective['interaction_window_s'])
            guidance,metrics=event_window_guidance(self.times,gap,window,
                required=self.editor.constraints.get('require_reference_contact',False) and not waypoint_route)
            residuals.extend(guidance)
            self.last_metrics=metrics
            forbidden=self.times < getattr(self.editor,'danger_start_s',0.)-.001
            residuals.extend(5*np.maximum(-gap[forbidden],0.))
        else:
            window = objective['interaction_window_s']
            mask = (self.times >= window[0]) & (self.times <= window[1])
            if objective['interaction'] == 'allowed':
                forbidden = self.times <= getattr(self.editor,'danger_start_s',0.)
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
        self.last_metrics.update(min_road_signed_margin_m=float(signed.min()),
            road_guidance_violation=bool(signed.min()<self.editor.constraints.get('official_margin_m',.1)-1e-8))
        residuals.extend((15 if self.editor.constraints.get('require_reference_contact') else 5)*np.maximum(self.editor.constraints.get('official_margin_m', .1)+.03-signed, 0.)/np.sqrt(8))
        first_other=None
        minimum_other=None
        overlap_count=0
        for observed, poses in self.obstacles:
            separation = signed_box_gap(xy[observed], yaw[observed], self.editor.target['size_m'], poses[:, :2], poses[:, 2], poses[:, 3:5])
            minimum_other=float(separation.min()) if minimum_other is None else min(minimum_other,float(separation.min()))
            contacts=self.times[observed][separation<0.]
            overlap_count+=len(contacts)
            if len(contacts):first_other=float(contacts[0]) if first_other is None else min(first_other,float(contacts[0]))
            residuals.extend((20 if self.editor.constraints.get('require_reference_contact') else 5)*np.maximum(.12-separation, 0.))
        self.last_metrics.update(min_other_signed_gap_m=minimum_other,
            other_overlap_count=overlap_count,first_other_contact_time_s=first_other,
            release_target_time_s=self.release_target_time)
        guidance,metrics=motion_guidance(self.editor,self.times,xy,yaw,initial_seed=(self.active_seed_xy,self.active_seed_yaw))
        residuals.extend(guidance)
        self.last_metrics=dict(getattr(self,'last_metrics',{}),**metrics)
        contact_time = self.last_metrics.get('first_reference_contact_time_s')
        if self.last_metrics.get('window_reference_contact') and contact_time is not None:
            if window[0] - .001 <= contact_time <= window[1] + .001:
                index = int(np.searchsorted(self.times, contact_time))
                prefix = slice(None, index + 1)
                prefix_motion = metrics['motion_guidance_violation_count']
                if prefix_motion and index >= 2:
                    _, prefix_metrics = motion_guidance(
                        self.editor, self.times[prefix], xy[prefix], yaw[prefix],
                        initial_seed=(self.active_seed_xy, self.active_seed_yaw))
                    prefix_motion = prefix_metrics['motion_guidance_violation_count']
                self.last_metrics['contact_prefix'] = {
                    'time_s': contact_time,
                    'xy_m': xy[index].tolist(),
                    'yaw_rad': float(yaw[index]),
                    'speed_mps': float(state[index, 0]),
                    'acceleration_mps2': float(state[index, 1]),
                    'steering_rad': float(state[index, 2]),
                    'road_margin_m': float(signed.reshape(-1, 8)[prefix].min()),
                    'third_party_clear': first_other is None or first_other > contact_time,
                    'motion_violations': prefix_motion,
                }
        self.last_metrics['geometry_motion_guidance_feasible']=bool(
            not self.last_metrics['road_guidance_violation'] and not overlap_count and
            not metrics['motion_guidance_violation_count'] and
            (waypoint_route or not self.editor.constraints.get('require_reference_contact')
             or self.last_metrics.get('window_reference_contact')))
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
        for braking in (-1., -3., -4.5, .5):
            values = self.initial.copy()
            values[:self.n] = braking
            yield values.clip(self.lower, self.upper)
        for turn in (-2., 2.):
            values = self.initial.copy()
            values[self.n:] += turn
            yield values.clip(self.lower, self.upper)
        for acceleration in (-4.5,-2.,1.5,2.5):
            for onset_fraction in (.25,.5,.7):
                values=self.initial.copy()
                onset=min(self.n-1,int(self.n*onset_fraction))
                duration=max(1,(self.n-onset)//2)
                values[onset:onset+duration]=acceleration
                yield values.clip(self.lower,self.upper)

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
            record['search_metrics']=dict(self.last_metrics)
            record['initial_pose_adjustment']=self.active_pose
            if self.search_phase:
                record['search_phase'] = self.search_phase
            records.append((float(residual @ residual), values.copy(), record))
            return residual

        # A timed route needs a full-horizon fit across all of its waypoints.
        staged = (self.editor.constraints.get('require_reference_contact', False)
                  and not self.arguments['objective'].get('waypoints')
                  and not self.arguments.get('preserve_outside_window')
                  and not self.editor.constraints.get('stationary_target'))
        self.search_phase = 'approach' if staged else None
        self.continuation_source = None
        starts = list(self.starts())
        if len(self.pose_specs)>1:
            count=min(len(starts),max(4,24//len(self.pose_specs)))
            starts=starts[:count]
        # Spend at most half on diverse rollouts; retain enough evaluations for
        # actual numerical steps instead of restarting a high-dimensional fit
        # in each tiny pose branch.
        seed_limit=min(limit,max(len(self.pose_specs),limit//2))
        def _eval_batch_parallel(batch_items):
            """Run pure CPU simulate+residual for independent (spec, start) pairs in parallel via fork."""
            import os, multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            n_workers = min(6, max(1, (os.cpu_count() or 1) // 2), len(batch_items))
            if n_workers <= 1 or 'fork' not in multiprocessing.get_all_start_methods():
                out = []
                for sp, vals in batch_items:
                    self.activate(sp)
                    ctrls = self.controls(vals)
                    arrs, st = self.simulate(ctrls)
                    res = self.residual(arrs, st)
                    out.append((sp, vals, ctrls, arrs, st, res, dict(self.last_metrics)))
                return out
            global _FORK_SEARCH_CTX
            _FORK_SEARCH_CTX = self
            try:
                ctx = multiprocessing.get_context('fork')
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                    return list(pool.map(_fork_eval_one, batch_items))
            finally:
                _FORK_SEARCH_CTX = None

        for spec_index,spec in enumerate(self.pose_specs):
            self.activate(spec)
            branch_start=len(records)
            branch_limit=branch_start+(seed_limit-branch_start)//(len(self.pose_specs)-spec_index)
            def branch_fun(values):
                if len(records)>=branch_limit:raise BudgetReached
                return fun(values)
            branch_starts = starts
            if getattr(self, 'waypoints', None):
                branch_starts = [starts[0], self.waypoint_start(), *starts[1:]]
            capped_starts = branch_starts[:max(0, branch_limit - len(records))]
            if len(capped_starts) >= 3:
                batch_results = _eval_batch_parallel([(spec, st) for st in capped_starts])
                for sp, vals, ctrls, arrs, st, res, metrics in batch_results:
                    if len(records) >= branch_limit or len(records) >= limit:
                        break
                    self.activate(sp)
                    self.last_metrics = metrics
                    if reserve is not None:
                        reserve(ctrls)
                    record = evaluate(ctrls, arrs, st, float(res @ res))
                    record['search_metrics'] = metrics
                    record['initial_pose_adjustment'] = self.active_pose
                    if self.search_phase:
                        record['search_phase'] = self.search_phase
                    records.append((float(res @ res), vals.copy(), record))
            else:
                for start in capped_starts:
                    if len(records)>=branch_limit:break
                    branch_fun(start)
            # Keep the current pose fixed. Derive release proposals from freshly
            # evaluated rollouts in this branch, never from another attack run.
            contact_records=[item for item in records[branch_start:]
                if item[2].get('search_metrics',{}).get('window_reference_contact')]
            if contact_records and not self.arguments.get('preserve_outside_window'):
                _,values,record=min(contact_records,key=lambda item:item[0])
                metrics=record['search_metrics']
                guesses=release_starts(self.control_times,self.editable,values,
                    metrics['release_target_time_s'],metrics.get('first_other_contact_time_s'))
                for guess in guesses:
                    if len(records)>=branch_limit:break
                    branch_fun(self.route_start(guess))
                    if len(records)>=branch_limit:break
                    branch_fun(guess.clip(self.lower,self.upper))
            branch_records=records[branch_start:]
            if not branch_records:continue
        if not records:return []
        if staged:
            from staged_control_search import continue_contact_prefixes
            if continue_contact_prefixes(self, records, fun, limit, BudgetReached):
                return sorted(records, key=lambda record: record[0])
            self.search_phase = 'approach_refinement'
        best=min(records,key=lambda item:(
            not item[2].get('search_metrics',{}).get('geometry_motion_guidance_feasible',False),item[0]))
        finalists=[best]
        contacts=[item for item in records if item[2].get('search_metrics',{}).get('window_reference_contact')]
        if contacts:
            contact=min(contacts,key=lambda item:item[0])
            if contact[2]['attempt_id']!=best[2]['attempt_id']:finalists.append(contact)
        for index,(_,start,record) in enumerate(finalists):
            if len(records)>=limit:break
            self.activate(record['initial_pose_adjustment'])
            groups,bounds=refinement_groups(start,self.lower,self.upper,self.n)
            if not groups:continue
            local_limit=len(records)+(limit-len(records))//(len(finalists)-index)
            def reduced_fun(offsets):
                if len(records)>=local_limit:raise BudgetReached
                values=start.copy()
                for group,offset in zip(groups,offsets):values[group]+=offset
                return fun(values.clip(self.lower,self.upper))
            try:
                least_squares(reduced_fun,np.zeros(len(groups)),bounds=(bounds[:,0],bounds[:,1]),
                    ftol=1e-4,xtol=1e-4,gtol=1e-4,max_nfev=20)
            except BudgetReached:
                pass
        return sorted(records, key=lambda record: record[0])
