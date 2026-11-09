"""Preserve the recorded body-centre velocity when starting a reference bicycle.

Recorded position and heading need not imply zero rear-axle lateral velocity.
Carry that bounded measured residual into the initial state and relax it smoothly
across one native sample interval. This is a reference reconstruction assumption,
not a calibrated tyre model or physical execution certificate.
"""
import math
import numpy as np
from vehicle_trajectory import sample_xy


def initial_state(times, xy, yaw, size, boundary, acceptance):
    centre = sample_xy(times, xy, [boundary])[0]
    angle = float(np.interp(boundary, times, np.unwrap(yaw)))
    before = max(float(times[0]), boundary - .1)
    if before >= boundary:
        raise ValueError('Initial bicycle state needs recorded motion before the boundary')
    velocity = (centre - sample_xy(times, xy, [before])[0]) / (boundary - before)
    before_angle = float(np.interp(before, times, np.unwrap(yaw)))
    yaw_rate = (angle - before_angle) / (boundary - before)
    forward = np.array([math.cos(angle), math.sin(angle)])
    left = np.array([-math.sin(angle), math.cos(angle)])
    speed = float(velocity @ forward)
    wheelbase = .6 * float(size[0])
    slip = float(velocity @ left) - wheelbase / 2 * yaw_rate
    if speed < -acceptance['attack_reverse_tolerance_mps']:
        raise ValueError('unsupported_initial_state: recorded motion is reversing')
    if abs(slip) > acceptance['attack_lateral_slip_mps']:
        raise ValueError('unsupported_initial_state: recorded rear-axle lateral slip exceeds policy')
    steering = math.atan(wheelbase * yaw_rate / max(speed, .1))
    if abs(steering) > min(.55, acceptance['attack_steering_rad']):
        raise ValueError('unsupported_initial_state: recorded steering exceeds generator range')
    return dict(rear_xy=centre-wheelbase/2*forward, yaw=angle, speed=max(0., speed),
                steering=steering, lateral_speed_mps=slip, boundary_s=float(boundary),
                relaxation_s=float(np.median(np.diff(times))))


def initial_lateral_velocity(state, time):
    phase = float(np.clip((time-state['boundary_s'])/state['relaxation_s'], 0., 1.))
    smooth = phase**3 * (10. + phase * (-15. + 6. * phase))
    return state['lateral_speed_mps'] * (1. - smooth)
