"""Shared ego history is defined on the native WorldEngine frame grid."""

from statistics import median


WE_TICK_S = 0.05
TAKEOVER_TIME_S = 1.5
MAX_NATIVE_GRID_JITTER_S = 0.005


def native_sample_rate(times):
    """Return the recorded frame interval in WorldEngine ticks."""
    return round(median(float(b - a) for a, b in zip(times, times[1:])) / WE_TICK_S)


def shared_history_end_index(times):
    """Find the last shared pose using nominal time and recorded frame order."""
    if len(times) < 2:
        return 0
    rate = native_sample_rate(times)
    if rate < 1:
        raise ValueError('Scene sampling is faster than WE native base interval')
    takeover_tick = round(TAKEOVER_TIME_S / WE_TICK_S)
    return min(len(times) - 1, takeover_tick // rate)
