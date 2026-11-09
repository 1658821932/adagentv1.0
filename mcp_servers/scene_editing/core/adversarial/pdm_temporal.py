"""PDM takeover timing and offline recorded-actor geometry.

Planner inputs remain native causal snapshots. Actor interpolation is evaluation
truth only, with explicit missing-pair coverage; it must not feed PlannerInput.
"""
from bisect import bisect_left, bisect_right
import math
from statistics import median

try:
    from .acceptance_policy import TIME_ROUNDING_S
except ImportError:
    from acceptance_policy import TIME_ROUNDING_S


def align_takeover(timestamps, history_s, tolerance_s=TIME_ROUNDING_S):
    requested = timestamps[0] + round(history_s * 1e6)
    following = bisect_left(timestamps, requested)
    if following < len(timestamps) and timestamps[following]-requested <= round(tolerance_s*1e6):
        index, actual = following, timestamps[following]
        mode = 'native_at_or_after_requested_boundary'
    else:
        index, actual = bisect_right(timestamps, requested)-1, requested
        mode = 'causal_extrapolation_from_previous_observation'
    return dict(index=index, takeover_time_us=actual, requested_time_us=requested,
                alignment_delay_s=(actual-requested)/1e6,
                extrapolation_s=(actual-timestamps[index])/1e6, mode=mode)


class RecordedActorReplay:
    """Interpolate the same recorded track, never future-only identities."""
    def __init__(self, times, observations):
        self.times = list(times)
        self.observations = observations
        self.by_track = [{obj['track']:obj for obj in frame} for frame in observations]
        self.max_pair_gap_s = 1.1 * median(b-a for a,b in zip(times,times[1:]))

    def sample(self, time):
        if time < self.times[0]-1e-7 or time > self.times[-1]+1e-7:
            raise ValueError('Replay time is outside recorded coverage')
        index = min(max(bisect_right(self.times,time+1e-7)-1,0),len(self.times)-1)
        left = self.observations[index]
        coverage = dict(observation_frame=index, interpolated_tracks=0, held_missing_pairs=0)
        if index == len(self.times)-1 or abs(time-self.times[index]) <= 1e-7:
            return left,coverage
        right = self.by_track[index+1]
        gap = self.times[index+1]-self.times[index]
        weight = (time-self.times[index])/gap
        output = []
        for obj in left:
            following = right.get(obj['track'])
            if following is None or gap > self.max_pair_gap_s or following['label'] != obj['label']:
                output.append(obj)
                coverage['held_missing_pairs'] += 1
                continue
            item = dict(obj)
            for key in ('x','y','length','width','height'):
                item[key] = obj[key] + weight*(following[key]-obj[key])
            delta = math.atan2(math.sin(following['yaw']-obj['yaw']),math.cos(following['yaw']-obj['yaw']))
            item['yaw'] = obj['yaw'] + weight*delta
            output.append(item)
            coverage['interpolated_tracks'] += 1
        return output,coverage
