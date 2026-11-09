"""Forward-only kinematic bicycle generation and reference-trajectory tracking."""
import acceptance_policy as policy
import math

import numpy as np

from vehicle_initial_state import initial_state, initial_lateral_velocity
from vehicle_trajectory import sample_xy


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def simulate_motion(times, seed_xy, seed_yaw, dense_times, size, parameters, ego, ego_yaw,
                    observation_end_s=3., reference_xy=None, reference_yaw=None, acceptance=None):
    """Integrate rear-axle speed/steering; return body centers on both output grids.

    Wheelbase is estimated as 60% of box length, rear axle 30% behind its center.
    This is a reference model assumption, not a calibrated vehicle or execution.
    """
    effective = policy.resolve(acceptance)
    speed_limit = effective['generator_max_speed_mps']
    wheelbase = .6 * float(size[0])
    shift = wheelbase / 2
    boundary = float(observation_end_s)
    center = sample_xy(times, seed_xy, [boundary])[0]
    initial = initial_state(times, seed_xy, seed_yaw, size, boundary, effective)
    angle, speed = initial['yaw'], initial['speed']
    if speed > speed_limit+1e-6:
        raise ValueError('unsupported_initial_state: speed exceeds configured generator range')
    initial_speed = speed
    rear = initial['rear_xy'].copy()
    acceleration, steering = 0., initial['steering']
    initial_ego = sample_xy(times, ego, [boundary])[0]
    initial_ego_yaw = float(np.interp(boundary, times, ego_yaw))
    initial_left = float((center - initial_ego) @ np.array([-math.sin(initial_ego_yaw), math.cos(initial_ego_yaw)]))
    if reference_xy is not None:
        reference_speed = np.linalg.norm(np.gradient(reference_xy, dense_times, axis=0), axis=1)
    query = np.unique(np.r_[times, dense_times, boundary])
    positions, angles = [], []
    previous_time = boundary
    for time in query:
        if time <= boundary:
            positions.append(sample_xy(times, seed_xy, [time])[0])
            angles.append(float(np.interp(time, times, np.unwrap(seed_yaw))))
            continue
        steps = max(1, int(math.ceil((time - previous_time) / .025)))
        step = (time - previous_time) / steps
        for index in range(steps):
            now = previous_time + (index + .5) * step
            center = rear + shift * np.array([math.cos(angle), math.sin(angle)])
            if reference_xy is None:
                start = parameters['lateral_start_s']
                duration = parameters['lateral_duration_s']
                phase = float(np.clip((now - start) / duration, 0., 1.))
                smooth = phase**3 * (10. + phase * (-15. + 6. * phase))
                desired_left = initial_left * (1 - smooth)
                lateral_velocity = -initial_left * 30 * phase**2 * (1-phase)**2 / duration
                ego_position = sample_xy(times, ego, [now])[0]
                ego_angle = float(np.interp(now, times, ego_yaw))
                left = float((center - ego_position) @ np.array([-math.sin(ego_angle), math.cos(ego_angle)]))
                lookahead = parameters['lookahead_m']
                heading = ego_angle + math.atan2(lateral_velocity - max(speed, 1.) * (left-desired_left) / lookahead,
                                                 max(speed, .8))
                curvature = 2 * math.sin(wrap(heading-angle)) / lookahead
                desired_speed = initial_speed
                if now >= parameters['brake_start_s']:
                    desired_speed = initial_speed * parameters['turn_speed_fraction']
                if now >= parameters['stop_time_s']:
                    desired_speed = 0.
                deceleration = parameters['deceleration_mps2']
            else:
                target = sample_xy(dense_times, reference_xy, [now])[0]
                target_yaw = float(np.interp(now, dense_times, reference_yaw))
                target -= shift * np.array([math.cos(target_yaw), math.sin(target_yaw)])
                delta = target-rear
                longitudinal_error=float(delta @ np.array([math.cos(target_yaw),math.sin(target_yaw)]))
                lateral_error=-float(delta @ np.array([-math.sin(target_yaw),math.cos(target_yaw)]))
                lookahead=max(2.,speed*.8)
                heading=target_yaw-math.atan2(lateral_error,lookahead)
                curvature=2*math.sin(wrap(heading-angle))/lookahead
                desired_speed=max(0.,float(np.interp(now,dense_times,reference_speed))+.8*longitudinal_error)
                deceleration = 4.5
            desired_acceleration = float(np.clip((desired_speed-speed) / .6, -deceleration, 2.5))
            jerk_limit=parameters.get('jerk_limit_mps3',4.)
            acceleration += float(np.clip(desired_acceleration-acceleration, -jerk_limit*step, jerk_limit*step))
            desired_steering = math.atan(wheelbase * curvature)
            if parameters.get('steering_controller')=='front_axle_goal':
                lateral_gain=parameters.get('lateral_gain',1.5)
                heading_gain=parameters.get('heading_gain',1.5)
                if reference_xy is None:
                    if parameters.get('steering_controller')=='front_axle_goal':
                        desired_left=initial_left if now<parameters['lateral_start_s'] else 0.
                    front_error=left+shift*math.sin(angle-ego_angle)-desired_left
                    heading_error=wrap(ego_angle-angle)
                else:
                    front_error=lateral_error+wheelbase*math.sin(angle-target_yaw)
                    heading_error=wrap(target_yaw-angle)
                desired_steering=heading_gain*heading_error-math.atan2(lateral_gain*front_error,max(speed,0.)+1.)
            steering_limit = min(.55, math.atan(3.4 * wheelbase / max(speed**2, .1)))
            desired_steering = float(np.clip(desired_steering, -steering_limit, steering_limit))
            steering_rate=parameters.get('steering_rate_radps',.5)
            steering += float(np.clip(desired_steering-steering, -steering_rate*step, steering_rate*step))
            next_speed = max(0., min(speed_limit, speed + acceleration * step))
            if desired_speed == 0 and next_speed < .015:
                next_speed = 0.
            middle_speed = (speed + next_speed) / 2
            yaw_rate = middle_speed / wheelbase * math.tan(steering)
            middle_angle = angle + yaw_rate * step / 2
            rear += step * (middle_speed * np.array([math.cos(middle_angle), math.sin(middle_angle)])
                            + initial_lateral_velocity(initial, now) * np.array([-math.sin(middle_angle), math.cos(middle_angle)]))
            angle += yaw_rate * step
            speed = next_speed
        center = rear + shift * np.array([math.cos(angle), math.sin(angle)])
        positions.append(center.copy()); angles.append(angle)
        previous_time = float(time)
    positions = np.asarray(positions); angles = np.asarray(angles)
    return (sample_xy(query, positions, times), np.interp(times, query, angles),
            sample_xy(query, positions, dense_times), np.interp(dense_times, query, angles))


def stopping_distance(speed):
    distance=acceleration=0.
    step=.01
    while speed>.001:
        desired=max(-4.5,-speed/.6)
        acceleration+=float(np.clip(desired-acceleration,-4.*step,4.*step))
        following=max(0.,speed+acceleration*step)
        distance+=(speed+following)*step/2
        speed=following
    return distance


def motion_specs(times, prefix_end, arguments, search_round=0, feedback=None, parent_parameters=None):
    lower, upper = arguments['time_window_s']
    proposals=[]
    for fraction in (0.,.4,.7,1.):
        for duration in (1.5,2.5,3.5):
            for lookahead in (2.,3.5):
                for stop_delay in (0.,1.):
                    proposals.append(dict(lateral_start_s=float(prefix_end),lateral_duration_s=duration,
                        lookahead_m=lookahead,turn_speed_fraction=fraction,brake_start_s=float(prefix_end),
                        stop_time_s=float(min(times[-1]-.5,prefix_end+duration+stop_delay)),
                        deceleration_mps2=4.5,steering_rate_radps=.5,jerk_limit_mps3=4.))
    for parameters in proposals:
        if parameters['lookahead_m']==2.:
            parameters.update(steering_controller='front_axle_goal',lateral_gain=3.5,heading_gain=1.5,steering_rate_radps=.7)
    from scipy.stats import qmc
    for sample in qmc.LatinHypercube(d=8,seed=int(search_round)).random(48):
        start=prefix_end+min(2.,max(0.,upper-prefix_end-1.5))*sample[0]
        duration=1.+3.5*sample[1]
        proposals.append(dict(lateral_start_s=float(start),lateral_duration_s=float(duration),
            lookahead_m=float(1.5+3.5*sample[2]),turn_speed_fraction=float(sample[3]),
            brake_start_s=float(prefix_end+1.2*sample[4]),
            stop_time_s=float(min(times[-1]-.5,start+duration+2*sample[5])),
            deceleration_mps2=4.5,steering_rate_radps=float(.35+.35*sample[6]),
            jerk_limit_mps3=float(4.+2*sample[7])))
    for parameters in proposals:
        yield dict(model='bicycle.v1',parameters=parameters,search_phase='coarse')
    if feedback is None:
        return
    rng=np.random.default_rng(1000+int(search_round))
    def score(record):
        if record.get('status')!='evaluated':
            return float('inf')
        return (30*len(record['failed_checks'])+10*record['motion_excess']+
                record['goal']['dense']['position_error_m'])
    coordinates=('heading_gain','lateral_gain','stop_time_s','steering_rate_radps')
    bounds=((.5,5.),(.5,6.),(prefix_end+.4,min(float(times[-1])-.5,upper)),(.3,.7))
    steps=np.array([.3,.5,.25,.05])
    for iteration in range(104):
        if any(r.get('reference_screen_passed') and r.get('goal',{}).get('goal_met') for r in feedback):
            return
        seeds=[r for r in feedback if r.get('parameters',{}).get('steering_controller')=='front_axle_goal']
        if seeds:
            best=min(seeds,key=score)['parameters']
        else:
            best=dict(parent_parameters or proposals[0],steering_controller='front_axle_goal',lateral_gain=3.5,heading_gain=1.5)
        parameters=dict(best)
        parameters.setdefault('steering_rate_radps',.5)
        scale=1. if iteration<48 else .5
        if iteration%2==0:
            coordinate=(iteration//2)%4
            direction=1 if (iteration//8)%2==0 else -1
            update=np.zeros(4);update[coordinate]=direction*steps[coordinate]*scale
        else:
            update=rng.normal(size=4)*steps*scale
        for name,delta,(minimum,maximum) in zip(coordinates,update,bounds):
            parameters[name]=float(np.clip(parameters[name]+delta,minimum,maximum))
        yield dict(model='bicycle.v1',parameters=parameters,search_phase='local_refinement')


def check_bicycle(times, xy, yaw, size):
    wheelbase = .6 * float(size[0])
    rear = xy - wheelbase/2 * np.column_stack([np.cos(yaw), np.sin(yaw)])
    velocity = np.gradient(rear, times, axis=0)
    forward = velocity[:, 0] * np.cos(yaw) + velocity[:, 1] * np.sin(yaw)
    lateral = -velocity[:, 0] * np.sin(yaw) + velocity[:, 1] * np.cos(yaw)
    yaw_rate = np.gradient(np.unwrap(yaw), times)
    moving = forward > .25
    steering = np.zeros(len(times))
    steering[moving] = np.arctan(wheelbase*yaw_rate[moving]/forward[moving])
    for index in range(1, len(times)):
        if not moving[index]:
            steering[index] = steering[index-1]
    steering_rate=np.gradient(steering,times)
    metrics = dict(min_forward_speed_mps=float(forward.min()), max_lateral_slip_mps=float(abs(lateral).max()),
        max_steering_rad=float(abs(steering).max()),
        max_steering_rate_radps=float(abs(steering_rate).max()),
        max_stationary_yaw_rate_radps=float(abs(yaw_rate[~moving]).max()) if (~moving).any() else 0.)
    checks = dict(no_reverse=metrics['min_forward_speed_mps'] >= -policy.ATTACK_REVERSE_TOLERANCE_MPS,
        no_lateral_slide=metrics['max_lateral_slip_mps'] <= policy.ATTACK_LATERAL_SLIP_MPS,
        steering_angle=metrics['max_steering_rad'] <= policy.ATTACK_STEERING_RAD,
        steering_rate=metrics['max_steering_rate_radps'] <= policy.ATTACK_STEERING_RATE_RADPS,
        no_stationary_rotation=metrics['max_stationary_yaw_rate_radps'] <= policy.ATTACK_STATIONARY_YAW_RATE_RADPS)
    measurements=[('no_reverse',-forward,policy.ATTACK_REVERSE_TOLERANCE_MPS),
        ('no_lateral_slide',abs(lateral),policy.ATTACK_LATERAL_SLIP_MPS),
        ('steering_angle',abs(steering),policy.ATTACK_STEERING_RAD),
        ('steering_rate',abs(steering_rate),policy.ATTACK_STEERING_RATE_RADPS),
        ('no_stationary_rotation',np.where(moving,0.,abs(yaw_rate)),policy.ATTACK_STATIONARY_YAW_RATE_RADPS)]
    violations=[]
    for name,values,limit in measurements:
        if not checks[name]:
            index=int(np.argmax(values));violations.append(dict(check=name,time_s=float(times[index]),actual=float(values[index]),limit=float(limit)))
    return dict(passed=all(checks.values()), checks=checks, metrics=metrics,violations=violations,
                wheelbase_m=wheelbase, scope='Estimated kinematic bicycle reference; not physical execution')
