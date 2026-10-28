"""Recover per-track Gaussian indices from UFO's per-pixel source arrays."""

from __future__ import annotations


def actor_mask(source: dict):
    """Mirror the reference assemble_scene.py mask and row-major ordering."""
    import numpy as np

    required = ("means", "scales", "opacities", "C", "dynamic", "coverage", "tracks")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError("UFO source is missing: " + ", ".join(missing))
    means = source["means"]
    if means.ndim != 3 or means.shape[2] != 3:
        raise ValueError("UFO source means must have shape [height,width,3]")
    shape = means.shape[:2]
    if any(source[key].shape != shape for key in ("opacities", "dynamic", "coverage", "tracks")):
        raise ValueError("UFO source masks do not match the Gaussian grid")
    if source["scales"].shape != means.shape or source["C"].shape != (4, 4):
        raise ValueError("UFO source scales or camera shape differs from the reference build")
    camera_xyz = (means - source["C"][:3, 3]) @ source["C"][:3, :3]
    depth = camera_xyz[..., 2]
    valid = (np.isfinite(means).all(-1) & np.isfinite(source["scales"]).all(-1)
             & (depth > .2) & (depth < 400) & (source["opacities"] > .01))
    return valid & source["dynamic"] & (source["coverage"] < .5)


def actor_track_indices(source: dict, track_id: int):
    """Return positions within ``frames/*.pt['actors']`` for one SAM track."""
    import numpy as np
    if type(track_id) is not int or track_id <= 0:
        raise ValueError("SAM track_id must be a positive integer")
    actor = actor_mask(source)
    target = actor & (source["tracks"] == track_id)
    indices = np.flatnonzero(target[actor])
    if not len(indices):
        raise ValueError(f"SAM track {track_id} has no actor Gaussians in this frame")
    return indices, int(actor.sum())


def match_sam_track(source: dict, boxes: list[tuple[int, int, int, int]],
                    *, minimum_dominance: float = .7,
                    minimum_coverage: float = .1) -> tuple[int, float]:
    """Match target boxes on UFO's 520x352 grid to a dominant SAM track."""
    import numpy as np

    tracks = source["tracks"]
    if tracks.ndim != 2:
        raise ValueError("UFO track mask must be two-dimensional")
    height, width = tracks.shape
    counts: dict[int, int] = {}
    box_area = 0
    for left, top, right, bottom in boxes:
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("target box lies outside UFO source grid")
        box_area += (right - left) * (bottom - top)
        values, frequencies = np.unique(tracks[top:bottom, left:right], return_counts=True)
        for value, frequency in zip(values, frequencies):
            if value > 0:
                counts[int(value)] = counts.get(int(value), 0) + int(frequency)
    if not counts:
        raise ValueError("no SAM track overlaps the selected target boxes")
    if sum(counts.values()) / box_area < minimum_coverage:
        raise ValueError("SAM track coverage inside the selected target is too small")
    track_id, count = max(counts.items(), key=lambda item: item[1])
    dominance = count / sum(counts.values())
    if dominance < minimum_dominance:
        raise ValueError(f"SAM target binding is ambiguous: dominance={dominance:.3f}")
    return track_id, dominance
