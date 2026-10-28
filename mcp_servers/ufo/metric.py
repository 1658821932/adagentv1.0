"""Robust native-unit to metre calibration from matched static depth samples."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable


def fit_metres_per_native_unit(
    samples: Iterable[tuple[float, float]], *, minimum_samples: int = 100,
    maximum_median_relative_error: float = 0.15,
    maximum_p90_relative_error: float = .25,
) -> dict[str, float | int]:
    """Fit one scene scale from (reconstruction camera-z, MoGe camera-z metres).

    The caller must provide corresponding static pixels in the same camera
    frame. This fit cannot establish correctness of the source correspondences.
    """
    ratios = []
    pairs = []
    for native, metres in samples:
        native = float(native)
        metres = float(metres)
        if not (math.isfinite(native) and math.isfinite(metres)) or native <= 0 or metres <= 0:
            continue
        ratios.append(metres / native)
        pairs.append((native, metres))
    if len(pairs) < minimum_samples:
        raise ValueError(f"metric calibration needs at least {minimum_samples} valid static depth pairs")
    scale = statistics.median(ratios)
    errors = [abs(native * scale - metres) / metres for native, metres in pairs]
    median_error = statistics.median(errors)
    p90_error = sorted(errors)[min(len(errors)-1, math.ceil(len(errors)*.9)-1)]
    if median_error > maximum_median_relative_error:
        raise ValueError(f"metric calibration residual {median_error:.3f} exceeds {maximum_median_relative_error:.3f}")
    if p90_error > maximum_p90_relative_error:
        raise ValueError(f'metric calibration p90 residual {p90_error:.3f} exceeds {maximum_p90_relative_error:.3f}')
    return {
        "metres_per_native_unit": scale,
        "native_units_per_metre": 1.0 / scale,
        "valid_samples": len(pairs),
        "median_relative_error": median_error,
        "p90_relative_error": p90_error,
    }


def metric_offset_to_native_world(
    forward_left_up_metres: tuple[float, float, float],
    camera_to_world: list[list[float]],
    metres_per_native_unit: float,
) -> tuple[float, float, float]:
    """Convert one camera-local FLU metric offset into UFO world units."""
    if len(forward_left_up_metres) != 3 or any(
        not math.isfinite(float(value)) for value in forward_left_up_metres
    ):
        raise ValueError("metric offset must be three finite [forward, left, up] values")
    if not math.isfinite(metres_per_native_unit) or metres_per_native_unit <= 0:
        raise ValueError("metres_per_native_unit must be finite and positive")
    if len(camera_to_world) != 4 or any(len(row) != 4 for row in camera_to_world):
        raise ValueError("camera_to_world must be a 4x4 matrix")
    forward, left, up = map(float, forward_left_up_metres)
    camera = (-left, -up, forward)
    result = tuple(
        sum(float(camera_to_world[row][column]) * camera[column] for column in range(3))
        / metres_per_native_unit
        for row in range(3)
    )
    if any(not math.isfinite(value) for value in result):
        raise ValueError("camera_to_world rotation must be finite")
    return result
