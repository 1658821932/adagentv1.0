"""MTGS reference poses in reconstruction coordinates, never raw UTM altitude."""
from datetime import date, datetime
import hashlib
import pickle
from pathlib import Path

import numpy as np
from native_pickle import SceneDataUnpickler


class ReconstructionUnpickler(SceneDataUnpickler):
    def find_class(self, module, name):
        if (module, name) in {('datetime', 'date'), ('datetime', 'datetime')}:
            return date if name == 'date' else datetime
        if (module, name) == ('pyquaternion.quaternion', 'Quaternion'):
            from pyquaternion import Quaternion
            return Quaternion
        return super().find_class(module, name)


class ReconstructionCalibration:
    def __init__(self, poses, recon_translation, provenance):
        self.poses = np.asarray(poses, dtype=float).copy()
        self.recon_translation = np.asarray(recon_translation, dtype=float).copy()
        if (self.poses.ndim != 3 or self.poses.shape[1:] != (4, 4)
                or not len(self.poses) or not np.isfinite(self.poses).all()
                or not np.allclose(self.poses[:, 3], [0., 0., 0., 1.], atol=1e-8)):
            raise ValueError('Invalid MTGS reconstruction reference poses')
        rotations = self.poses[:, :3, :3]
        if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(rotations), 1., atol=1e-4)):
            raise ValueError('MTGS reconstruction pose rotations must be proper rigid rotations')
        if self.recon_translation.shape != (3,) or not np.isfinite(self.recon_translation).all():
            raise ValueError('Invalid MTGS reconstruction origin')
        self.provenance = dict(provenance, coordinate_frame='reconstruction',
            recon2world_translation=self.recon_translation.tolist(), reference_pose_count=len(self.poses))

    def verify_origin(self, translation):
        if not np.allclose(translation, self.recon_translation, atol=1e-6, rtol=0):
            raise ValueError('Reconstruction reference and Gaussian checkpoint origins differ')

    def nearest_index(self, xy):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('Reference lookup requires finite reconstruction XY')
        return int(np.argmin(np.linalg.norm(self.poses[:, :2, 3] - xy, axis=1)))

    def pose_at(self, xy, heading, road_map=None):
        """Native MTGS nearest-XY Z/tilt, requested path XY/yaw. For edited
        paths follow the local road with the reference's measured clearance."""
        if not np.isfinite(heading):
            raise ValueError('Reference lookup requires a finite heading')
        pose = self.poses[self.nearest_index(xy)].copy()
        if road_map is not None:
            base = road_map.height_with_local_plane(*pose[:2, 3])['height']
            target = road_map.height_with_local_plane(*xy)['height']
            pose[2, 3] += target - base
        yaw = np.arctan2(pose[1, 0], pose[0, 0])
        delta = float(heading - yaw)
        c, s = np.cos(delta), np.sin(delta)
        pose[:3, :3] = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ pose[:3, :3]
        pose[:2, 3] = xy
        return pose


def recorded_pose_at(calibration, xy, heading, road_map):
    """Render a recorded comparison even where road measurements are absent.

    Native MTGS still has a measured reconstruction pose there; report road
    height as unknown rather than pretending the surface has been measured.
    Edited candidates continue to require pose_at(..., road_map).
    """
    try:
        ground = road_map.height_with_local_plane(*xy)
        return calibration.pose_at(xy, heading, road_map), ground
    except ValueError as error:
        return calibration.pose_at(xy, heading), dict(height=None,
            source='native_reconstruction_reference',road_height_status='unavailable',
            reason=str(error),reference_index=calibration.nearest_index(xy))


def load_reconstruction_calibration(scene_dir, frames, *, asset_id=None):
    path = Path(scene_dir).resolve() / 'source_asset' / 'video_scene_dict.pkl'
    try:
        with path.open('rb') as stream:
            videos = ReconstructionUnpickler(stream).load()
    except (OSError, pickle.UnpicklingError, EOFError, ImportError) as error:
        raise ValueError('MTGS reconstruction calibration unavailable: ' + str(path) + ': ' + str(error)) from error
    if not isinstance(videos, dict) or not videos:
        raise ValueError('Expected a nonempty MTGS video_scene_dict')
    requested = {str(frame['token']) for frame in frames}
    candidates = []
    for key, video in videos.items():
        if not isinstance(video, dict) or not isinstance(video.get('frame_infos'), list):
            continue
        identity = str(video.get('video_token', key))
        if asset_id and identity != asset_id and not identity.startswith(asset_id + '-'):
            continue
        # The official converter retains all reconstruction reference poses;
        # "skipped" is an image/training flag, not a missing pose marker.
        rows = video['frame_infos']
        count = len({str(row['token']) for row in rows} & requested)
        if count:
            candidates.append((count, identity, video, rows))
    if not candidates:
        raise ValueError('No reconstruction video is bound to the requested scene frame tokens and asset')
    best = max(candidate[0] for candidate in candidates)
    candidates = [candidate for candidate in candidates if candidate[0] == best]
    if len(candidates) != 1:
        raise ValueError('Ambiguous reconstruction video binding for this scene')
    count, identity, video, rows = candidates[0]
    # Preserve the road-block reference cloud used by native nearest-XY lookup.
    # Scene frames beyond the reconstruction span are not timestamp-extrapolated.
    return ReconstructionCalibration([row['ego2global'] for row in rows], video['global2world_translation'],
        dict(version='mtgs_reconstruction_reference.v1', source='video_scene_dict.frame_infos.ego2global',
            source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            video_token=identity, exact_scene_token_count=count, scene_frame_count=len(frames),
            scene_tokens_without_exact_reference=len(requested) - count,
            lookup='nearest reconstruction XY; no timestamp extrapolation',
            reference_start_timestamp_us=int(rows[0]['timestamp']),
            reference_end_timestamp_us=int(rows[-1]['timestamp'])))
