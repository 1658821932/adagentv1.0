"""Physical-time scoring windows: never pad, stretch or extrapolate source observations."""
from __future__ import annotations

import numpy as np


def scoring_times(source_times, start: float, horizon: float, interval: float = 0.1):
    source = np.asarray(source_times, dtype=float)
    if source.ndim != 1 or len(source) < 2 or not np.isfinite(source).all() or np.any(np.diff(source) <= 0):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    if not np.isfinite([start, horizon, interval]).all() or horizon <= 0 or interval <= 0:
        raise ValueError("Scoring times must be finite with positive horizon and interval")
    steps = int(round(horizon / interval))
    if not np.isclose(steps * interval, horizon):
        raise ValueError("Scoring horizon must be a positive integer number of intervals")
    times = start + np.arange(steps + 1) * interval
    if times[0] < source[0] - 1e-8 or times[-1] > source[-1] + 1e-8:
        raise ValueError(f"Insufficient observed future: need {times[0]:.3f}..{times[-1]:.3f}s, "
                         f"have {source[0]:.3f}..{source[-1]:.3f}s; extrapolation is forbidden")
    return times


def interpolate_track(state: dict, source_times, target_times) -> dict:
    source = np.asarray(source_times, dtype=float)
    target = np.asarray(target_times, dtype=float)
    if target.min() < source[0] or target.max() > source[-1]:
        raise ValueError("Track extrapolation is forbidden")
    result = {}
    valid = np.asarray(state.get("valid", np.ones(len(source))), dtype=bool).reshape(-1)
    left = np.clip(np.searchsorted(source, target, side="right") - 1, 0, len(source) - 1)
    right = np.clip(np.searchsorted(source, target, side="left"), 0, len(source) - 1)
    if not (valid[left] & valid[right]).all():
        raise ValueError("Scoring window crosses invalid tracked-object observations")
    for key in ("position", "heading", "velocity", "length", "width", "height"):
        values = np.asarray(state[key], dtype=float)
        if len(values) != len(source) or not np.isfinite(values).all():
            raise ValueError(f"Invalid track field: {key}")
        if key == "heading":
            values = np.unwrap(values.reshape(-1))
        if values.ndim == 1:
            result[key] = np.interp(target, source, values)
        else:
            result[key] = np.stack([np.interp(target, source, values[:, i])
                                    for i in range(values.shape[1])], axis=1)
    return result
