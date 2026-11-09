"""Sampled contact proxies and entry dynamics, without event pass thresholds."""
import numpy as np


def rectangle_overlap(relative, angle, target_length, target_width, ego_length, ego_width):
    """Separating-axis test in both rectangles' axes, including touching edges."""
    x, y = np.asarray(relative).T
    c, s = np.cos(angle), np.sin(angle)
    return ((np.abs(x) <= ego_length/2+target_length/2*np.abs(c)+target_width/2*np.abs(s)) &
        (np.abs(y) <= ego_width/2+target_length/2*np.abs(s)+target_width/2*np.abs(c)) &
        (np.abs(c*x+s*y) <= target_length/2+ego_length/2*np.abs(c)+ego_width/2*np.abs(s)) &
        (np.abs(-s*x+c*y) <= target_width/2+ego_length/2*np.abs(s)+ego_width/2*np.abs(c)))


def entry_dynamics(times, relative, closing, gap, before, after, proxy_half_width=1):
    times, relative, closing, gap = map(np.asarray, (times, relative, closing, gap))
    active = (times >= before) & (times <= after)
    ids = np.flatnonzero(active)
    left = relative[:, 1]
    entries = np.flatnonzero((np.abs(left) <= proxy_half_width) & (relative[:, 0] > 0) & (times >= before))
    transition = None
    if len(ids):
        initial, final = left[ids[0]], left[ids[-1]]
        if abs(final-initial) > .05:
            progress = (left-initial)/(final-initial)
            ten = np.flatnonzero(active & (progress >= .1-1e-9))
            if len(ten):
                ninety = np.flatnonzero(active & (np.arange(len(times)) >= ten[0]) & (progress >= .9-1e-9))
                if len(ninety):
                    transition = dict(start_s=float(times[ten[0]]), end_s=float(times[ninety[0]]),
                        duration_s=float(times[ninety[0]]-times[ten[0]]))
    entry = matching = None
    if len(entries):
        i = int(entries[0])
        entry = dict(time_s=float(times[i]), from_declared_edit_start_s=float(times[i]-before),
            relative_forward_m=float(relative[i, 0]), relative_left_m=float(left[i]),
            projected_body_gap_m=float(gap[i]), closing_speed_mps=float(closing[i]))
        matched = np.flatnonzero((np.arange(len(times)) >= i) & (closing <= .1))
        if len(matched):
            j = int(matched[0])
            matching = dict(time_s=float(times[j]), seconds_after_entry=float(times[j]-times[i]),
                projected_body_gap_m=float(gap[j]), closing_speed_mps=float(closing[j]))
    return dict(lateral_first_passage_10_to_90=transition, first_proxy_entry=entry,
        first_nonclosing_after_entry=matching,
        scope='10%-90% is first passage toward the declared boundary lateral pose, not an action deadline; '
              'entry uses ego-width path proxy, not lanes. Nonclosing threshold is 0.1m/s, descriptive only. '
              'Already-inside and nonmonotone paths require reading the full trajectory.')
