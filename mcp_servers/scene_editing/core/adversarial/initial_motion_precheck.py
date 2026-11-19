"""Raw annotation diagnostics and the backend's generator initial-state check."""
import math
import acceptance_policy as policy


def initial_ego_box(frame, box, first_frame):
    transform = frame.get('lidar2global')
    origin = first_frame.get('ego2global')
    if transform is None or origin is None:
        return box
    lidar_yaw = math.atan2(transform[1][0], transform[0][0])
    origin_yaw = math.atan2(origin[1][0], origin[0][0])
    c, s = math.cos(lidar_yaw), math.sin(lidar_yaw)
    global_xy = [transform[0][3]+c*box[0]-s*box[1],
                 transform[1][3]+s*box[0]+c*box[1]]
    delta = [global_xy[0]-origin[0][3], global_xy[1]-origin[1][3]]
    c, s = math.cos(origin_yaw), math.sin(origin_yaw)
    return [c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1], box[2],
            *box[3:6], lidar_yaw+box[6]-origin_yaw]


def initial_motion_precheck(frames, track_id, acceptance=None):
    effective = policy.resolve(acceptance)
    if len(frames) < 2:
        return dict(track_id=track_id, passed=True, errors=[],
                    scope='Initial motion requires at least two frames; no incompatibility inferred')
    samples = []
    for frame in frames:
        anns = frame.get('anns', {})
        if track_id not in anns.get('track_tokens', []):
            return dict(track_id=track_id, passed=False, errors=['Target track is missing from the initial frames'])
        index = anns['track_tokens'].index(track_id)
        box = initial_ego_box(frame, anns['gt_boxes'][index], frames[0])
        samples.append((float(frame['timestamp']), box))

    origin = samples[0][0]
    intervals = sorted(right-left for (right,_),(left,_) in zip(samples[1:],samples[:-1]))
    time_scale = 1e6 if intervals and intervals[len(intervals)//2] > 1000 else 1
    samples = [((time-origin)/time_scale, box) for time, box in samples]
    yaws = [samples[0][1][6]]
    for _, box in samples[1:]:
        delta = math.atan2(math.sin(box[6]-yaws[-1]), math.cos(box[6]-yaws[-1]))
        yaws.append(yaws[-1]+delta)

    def sample(boundary):
        for index in range(1, len(samples)):
            right_time, right = samples[index]
            left_time, left = samples[index-1]
            if boundary <= right_time:
                ratio = (boundary-left_time)/(right_time-left_time)
                position = [left[axis]+(right[axis]-left[axis])*ratio for axis in (0, 1)]
                yaw = yaws[index-1]+(yaws[index]-yaws[index-1])*ratio
                return position, yaw
        raise ValueError('Invalid initial sample boundary')

    violations = []
    measurement = {}
    for index in range(1, len(samples)):
        boundary = samples[index][0]
        # Differentiate over the native interval, not 0.1 s: at 2 Hz annotation
        # precision, a 0.1 s baseline amplifies position noise tenfold.
        before = samples[index-1][0]
        center, angle = sample(boundary)
        previous, previous_angle = sample(before)
        span = boundary-before
        velocity = [(center[axis]-previous[axis])/span for axis in (0, 1)]
        yaw_rate = (angle-previous_angle)/span
        forward = (math.cos(angle), math.sin(angle))
        left = (-math.sin(angle), math.cos(angle))
        speed = velocity[0]*forward[0] + velocity[1]*forward[1]
        wheelbase = .6*float(samples[0][1][3])
        slip = velocity[0]*left[0] + velocity[1]*left[1] - wheelbase/2*yaw_rate
        steering = math.atan(wheelbase*yaw_rate/max(speed, .1))
        measurement = dict(forward_speed_mps=speed, rear_axle_lateral_speed_mps=slip, steering_rad=steering)
        errors = []
        # A stopped vehicle has no meaningful steering angle: dividing yaw rate by a
        # near-zero speed makes the derived angle explode. Only a rolling vehicle can
        # violate these limits.
        # 0.05-0.1 m/s sits inside annotation precision at 2 Hz; a parked car jitters
        # by centimetres per frame. Treat anything under 0.3 m/s as stopped.
        stopped = abs(speed) <= PRECHECK_STILL_SPEED_MPS
        reverse_limit = PRECHECK_STILL_SPEED_MPS
        if speed < -reverse_limit:
            errors.append('unsupported_initial_state: recorded motion is reversing')
        if not stopped and abs(slip) > effective['attack_lateral_slip_mps']:
            errors.append('unsupported_initial_state: recorded rear-axle lateral slip exceeds policy')
        if not stopped and abs(steering) > min(.55, effective['attack_steering_rad']):
            errors.append('unsupported_initial_state: recorded steering exceeds generator range')
        if errors:
            violations.append(dict(time_s=boundary, forward_speed_mps=speed,
                rear_axle_lateral_speed_mps=slip, steering_rad=steering, errors=errors))
    errors = list(dict.fromkeys(error for violation in violations for error in violation['errors']))
    result = dict(track_id=track_id, passed=not violations, errors=errors, violations=violations,
                  scope='All observed intervals; same bicycle initial-state limits as tool05')
    result.update(measurement)
    if violations:
        result.update(first_violation_time_s=violations[0]['time_s'],
                      forward_speed_mps=violations[0]['forward_speed_mps'],
                      rear_axle_lateral_speed_mps=violations[0]['rear_axle_lateral_speed_mps'],
                      steering_rad=violations[0]['steering_rad'])
    return result


PRECHECK_STILL_SPEED_MPS = 0.3

def initial_state_precheck(editor):
    if editor.target['operation'] != 'edit_existing':
        return None
    if len(editor.times) < 2:
        return None
    # Selection CSVs already use the fixed initial-ego frame. Applying each
    # current ego2global again invents target motion as the ego moves/turns.
    # Check the inherited boundary with the actual generator, not the editable
    # future of the original recording. Generated candidates are checked later.
    from vehicle_initial_state import initial_state
    effective = policy.resolve(editor.constraints.get('acceptance_policy'))
    boundary = float(editor.edit_start_s)
    result = dict(track_id=editor.target['track_id'], boundary_s=boundary,
        coordinate_frame='initial_ego',
        scope='Inherited state at the edit boundary; editable future is checked after construction')
    try:
        state = initial_state(editor.times, editor.seed_xy, editor.seed_yaw,
                              editor.target['size_m'], boundary, effective)
        result.update(forward_speed_mps=state['speed'],
            rear_axle_lateral_speed_mps=state['lateral_speed_mps'],
            steering_rad=state['steering'])
        if state['speed'] > effective['generator_max_speed_mps'] + 1e-6:
            raise ValueError('unsupported_initial_state: speed exceeds configured generator range')
    except ValueError as exc:
        if not str(exc).startswith('unsupported_initial_state:'):
            raise
        result.update(passed=False, errors=[str(exc)])
    else:
        return None
    return dict(stop_reason='unsupported_initial_state', reason='; '.join(result['errors']),
        scope='Object initial state is incompatible with the trajectory generator; choose another object, add a vehicle or change scene.',
        global_infeasibility_proven=False, measurements=result)
