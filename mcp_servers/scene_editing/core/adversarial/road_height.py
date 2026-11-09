"""Road surface height and orientation from the scene's own road_height_map asset.

The MTGS reconstruction (asset frame) is the visual ground truth for rendering.
UTM annotation z drifts from the reconstructed road along the path (f289 drifts
~0.9 m over 50 m), so anything placed by UTM z sinks into the rendered road. The
scene ships source_asset/road_height_map/{road_height_map.npy, sim2.json} sampled
from the reconstruction; sim2 maps asset-frame xy to pixel via
pixel = scale * (R @ xy + translation).
"""
import functools
import json
import math
from pathlib import Path

import numpy as np


class RoadHeightMap:
    def __init__(self, heights, rotation, translation, scale):
        heights = np.asarray(heights, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        translation = np.asarray(translation, dtype=float)
        if heights.ndim != 2 or min(heights.shape) < 2 or not np.isfinite(heights).any():
            raise ValueError('Road height map must be a 2D grid with finite values')
        if rotation.shape != (2, 2) or not np.allclose(rotation @ rotation.T, np.eye(2), atol=1e-6):
            raise ValueError('sim2 rotation must be orthonormal')
        if translation.shape != (2,) or not np.isfinite(translation).all():
            raise ValueError('sim2 translation must be a finite 2-vector')
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('sim2 scale must be positive')
        self.heights = heights
        self.rotation = rotation
        self.translation = translation
        self.scale = float(scale)

    @classmethod
    def load(cls, scene_dir):
        folder = Path(scene_dir).resolve()/'source_asset'/'road_height_map'
        map_path, sim2_path = folder/'road_height_map.npy', folder/'sim2.json'
        if not map_path.is_file() or not sim2_path.is_file():
            raise ValueError(f'Scene has no road_height_map asset: {folder}')
        sim2 = json.loads(sim2_path.read_text())
        return cls(np.load(map_path), sim2['R'], sim2['t'], sim2['s'])

    def pixel(self, x, y):
        px, py = self.scale*(self.rotation @ np.array([float(x), float(y)]) + self.translation)
        return float(px), float(py)

    def height_at(self, x, y):
        """Bilinear road height at asset-frame xy; raises outside the map or on
        cells without road data (NaN padding)."""
        px, py = self.pixel(x, y)
        rows, cols = self.heights.shape
        if not (0 <= px <= cols-1 and 0 <= py <= rows-1):
            raise ValueError(f'Position ({x:.2f}, {y:.2f}) is outside the road height map')
        x0, y0 = int(math.floor(px)), int(math.floor(py))
        x1, y1 = min(x0+1, cols-1), min(y0+1, rows-1)
        fx, fy = px-x0, py-y0
        z = (self.heights[y0, x0]*(1-fx)*(1-fy) + self.heights[y0, x1]*fx*(1-fy)
             + self.heights[y1, x0]*(1-fx)*fy + self.heights[y1, x1]*fx*fy)
        if not math.isfinite(z):
            raise ValueError(f'No road height data at ({x:.2f}, {y:.2f})')
        return float(z)

    def height_with_local_plane(self, x, y, radius_m=.75, max_residual_m=.05):
        """Fill an enclosed small hole using nearby measured heights; never extrapolate."""
        direct = self.try_height_at(x, y)
        if direct is not None:
            return dict(height=direct, source='bilinear', sample_count=4, max_residual_m=0.)
        if not self.covers(x, y):
            raise ValueError('Target is outside the road height map')
        from shapely.geometry import MultiPoint, Point
        samples = []
        for dx in np.linspace(-radius_m, radius_m, 7):
            for dy in np.linspace(-radius_m, radius_m, 7):
                z = self.try_height_at(x+dx, y+dy)
                if z is not None:
                    samples.append((dx, dy, z))
        if len(samples) < 6 or not MultiPoint([v[:2] for v in samples]).convex_hull.contains(Point(0., 0.)):
            raise ValueError('Missing road height has insufficient surrounding measurements')
        values = np.asarray(samples)
        design = np.column_stack([values[:, :2], np.ones(len(values))])
        coef, _, rank, _ = np.linalg.lstsq(design, values[:, 2], rcond=None)
        residual = float(np.max(np.abs(design@coef-values[:, 2])))
        if rank < 3 or residual > max_residual_m:
            raise ValueError('Local road-height plane fit exceeds the residual limit')
        return dict(height=float(coef[2]), source='enclosed_local_plane', sample_count=len(values),
                    radius_m=radius_m, max_residual_m=residual)

    def try_height_at(self, x, y):
        try:
            return self.height_at(x, y)
        except ValueError:
            return None

    def covers(self, x, y):
        px, py = self.pixel(x, y)
        rows, cols = self.heights.shape
        return 0 <= px <= cols-1 and 0 <= py <= rows-1

    def placement_plane(self, x, y, length, width, yaw):
        """Fit the road plane under a vehicle footprint; returns center height,
        the 3x3 rotation aligning local +X-forward/+Z-up to the road, and the
        worst corner residual (road flatness over the footprint)."""
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        corners = np.array([(dx, dy) for dx in (length/2, -length/2) for dy in (width/2, -width/2)])
        xy = np.array([x, y]) + corners @ np.array([[c, s], [-s, c]])
        z = np.array([self.height_at(px, py) for px, py in xy])
        design = np.column_stack([xy, np.ones(len(xy))])
        (a, b, base), *_ = np.linalg.lstsq(design, z, rcond=None)
        residual = float(np.max(np.abs(z-(design @ [a, b, base]))))
        normal = np.array([-a, -b, 1.]); normal /= np.linalg.norm(normal)
        forward = np.array([c, s, 0.])
        x_axis = forward-normal*np.dot(forward, normal); x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        rotation = np.column_stack([x_axis, y_axis, normal])
        return dict(z_center=float(a*x+b*y+base), rotation=rotation,
                    pitch=float(math.asin(np.clip(x_axis[2], -1, 1))),
                    roll=float(math.asin(np.clip(y_axis[2], -1, 1))),
                    max_corner_residual_m=residual)


@functools.lru_cache(maxsize=4)
def read_recon_translation(scene_dir):
    """recon2world_translation (asset frame = UTM global - translation) from the
    scene checkpoint config. Loaded once per scene; mmap keeps Gaussian tensors
    on disk."""
    scene_dir = Path(scene_dir).resolve()
    checkpoints = list((scene_dir/'source_asset'/'background').glob('*.ckpt'))
    if len(checkpoints) != 1:
        raise ValueError(f'Expected exactly one background checkpoint in {scene_dir}')
    from render_ego_mtgs import validate_checkpoint_pickle
    validate_checkpoint_pickle(str(checkpoints[0]))
    import torch
    asset = torch.load(str(checkpoints[0]), map_location='cpu', mmap=True)
    translation = np.asarray(asset['background']['config']['recon2world_translation'], float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError('Checkpoint config has no finite recon2world_translation')
    return translation


def ego_height_reference(frames, road_map, recon_translation):
    """Per-frame vertical offset of the recorded ego origin above the road map,
    plus the frame-0 value as the calibration reference. The recon2world
    alignment is anchored near the scene start, so frame 0 reflects the physical
    mount height; later frames may drift with the reconstruction."""
    recon_translation = np.asarray(recon_translation, float)
    profile = []
    for i, frame in enumerate(frames):
        ego = np.asarray(frame['ego2global'], float)
        ex, ey = ego[0, 3]-recon_translation[0], ego[1, 3]-recon_translation[1]
        road_z = road_map.try_height_at(ex, ey)
        profile.append(None if road_z is None else float(ego[2, 3]-recon_translation[2]-road_z))
    covered = [v for v in profile if v is not None]
    if not covered or profile[0] is None:
        raise ValueError('Recorded ego start is outside the road height map')
    return dict(reference_m=profile[0], profile_m=profile,
                min_m=float(min(covered)), max_m=float(max(covered)))
