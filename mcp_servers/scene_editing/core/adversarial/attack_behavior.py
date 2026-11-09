"""Temporal interaction evidence against the recorded ego path, not map lane IDs."""
import numpy as np

from run_ego_reference import EGO_FRONT, EGO_REAR, EGO_WIDTH, polygon


def behavior_evidence(times, xy, yaw, ego, ego_yaw, size, arguments):
    lower, upper = arguments['time_window_s']
    mode = arguments['interaction_mode']
    relative = xy-ego
    forward = relative[:,0]*np.cos(ego_yaw)+relative[:,1]*np.sin(ego_yaw)
    left = -relative[:,0]*np.sin(ego_yaw)+relative[:,1]*np.cos(ego_yaw)
    angles = np.arctan2(np.sin(yaw-ego_yaw),np.cos(yaw-ego_yaw))
    gaps, contacts = [], []
    for i in range(len(times)):
        ego_center = ego[i]+(EGO_FRONT-EGO_REAR)/2*np.array([np.cos(ego_yaw[i]),np.sin(ego_yaw[i])])
        own = polygon(*ego_center,ego_yaw[i],EGO_FRONT+EGO_REAR,EGO_WIDTH)
        target = polygon(*xy[i],yaw[i],*size[:2])
        gaps.append(float(own.distance(target)))
        if own.intersection(target).area>1e-9:
            contacts.append(i)
    gaps = np.asarray(gaps)
    if mode == 'either':
        mode = 'contact' if contacts else 'near_miss'
    prefix_index = int(np.argmin(abs(times-3.)))
    initial_left = float(left[prefix_index])
    side = arguments['entry_side']
    side_ok = initial_left>1.05 if side=='left' else initial_left < -1.05 if side=='right' else abs(initial_left)>1.05
    entered = np.flatnonzero((times>3.) & (abs(left)<=.75))
    entry_time = None if not len(entered) else float(times[entered[0]])
    entry_ok = entry_time is not None and lower-1e-3 <= entry_time <= upper+1e-3
    half_extent = abs(np.cos(angles))*size[0]/2+abs(np.sin(angles))*size[1]/2
    front_gap = forward-half_extent-EGO_FRONT
    relation = front_gap>=.15 if mode=='near_miss' else forward>=(EGO_FRONT-EGO_REAR)/2
    eligible = ((times>=lower) & (times<=upper) & (abs(left)<=.75) &
                (abs(angles)<=.35) & relation)
    longest = 0.; best_start = best_end = None; start = None
    for i,good in enumerate(eligible):
        if good and start is None:
            start=i
        if start is not None and (not good or i==len(times)-1):
            end=i if good else i-1
            duration=float(times[end]-times[start])
            if duration>longest:
                longest=duration;best_start=start;best_end=end
            start=None
    event_mask=(times>=lower)&(times<=upper)
    contact_ok = (bool(contacts) and lower<=times[contacts[0]]<=upper) if mode=='contact' else not contacts and float(gaps.min())>=.15
    near_ok = True if mode=='contact' else bool(np.any(eligible & (gaps<=2.)))
    entry_front = False if not len(entered) else bool(relation[entered[0]])
    checks=dict(initial_side=bool(side_ok), entered_in_window=bool(entry_ok), entry_in_front=entry_front,
        sustained_aligned_relation=longest+1e-3>=arguments['hold_duration_s'],
        requested_interaction=bool(contact_ok), close_interaction=bool(near_ok))
    penalty=abs(left)+2*abs(angles)+np.maximum(-forward,0.)
    if mode=='contact' and not contacts:
        penalty+=2*gaps
    if mode=='near_miss':
        penalty+=np.maximum(.15-front_gap,0.)+np.maximum(gaps-2.,0.)
    indices=np.flatnonzero(event_mask)
    index=int(indices[np.argmin(penalty[event_mask])])
    error=float(penalty[index]+3*max(0.,arguments['hold_duration_s']-longest)+ (0 if contact_ok else 3))
    return dict(goal_met=all(checks.values()), interaction_mode=mode, checks=checks,
        failed_checks=[k for k,v in checks.items() if not v],position_error_m=error,
        measured_time_s=float(times[index]),actual_relative_forward_m=float(forward[index]),
        actual_relative_left_m=float(left[index]),actual_heading_error_rad=float(angles[index]),
        heading_error_limit_rad=.35,corridor_half_width_m=.75,initial_left_m=initial_left,entry_time_s=entry_time,
        entry_forward_to_ego_center_m=None if not len(entered) else float(forward[entered[0]]-(EGO_FRONT-EGO_REAR)/2),
        sustained_duration_s=longest,hold_interval_s=None if best_start is None else [float(times[best_start]),float(times[best_end])],
        first_contact_time_s=None if not contacts else float(times[contacts[0]]),min_clearance_m=float(gaps.min()),
        scope='Entry into +/-0.75m recorded-ego path corridor, heading within 0.35rad, sustained front relation; not official lane assignment or physical collision')
