"""Verify estimated metric units already established by the pose frontend.

This establishes a coordinate-unit contract, NOT dense geometry accuracy or
independent real-world distance certification. Depth-fit validation is separate.
"""
from __future__ import annotations

import json
from pathlib import Path


def verify_frontend_units(workspace: Path, asset_root: Path) -> dict:
    import numpy as np
    from .metric_scene import digest
    metadata_path = workspace / 'input/cameras/scenes/video/metadata.json'
    poses_path = workspace / 'input/cameras/global_aligned/video/omega_pose_override.npz'
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('camera_basis') != 'OpenCV: right,down,forward' or 'MoGe-2/GCA' not in metadata.get('metric_scale_source', ''):
        raise ValueError('Frontend metadata does not establish estimated metric OpenCV camera units')
    with np.load(poses_path, allow_pickle=False) as source:
        if str(source['coordinate_frame']) != 'global_metric' or str(source['metric_scale_source']) != 'moge2_gca_plus_overlap_camera_se3':
            raise ValueError('Unsupported frontend metric coordinate contract')
        ids = source['frame_ids'].copy()
        cameras = source['camera_ids'].astype(str)
        upstream = source['omega_camera_to_world_global_metric'].astype(np.float64)
        upstream_K = source['predicted_intrinsics_ufo'].astype(np.float64)
    with np.load(asset_root / 'assets' / workspace.name / 'cameras.npz', allow_pickle=False) as target:
        actual, actual_K = target['C'].astype(np.float64), target['K'].astype(np.float64)
    if upstream_K.shape != (40,3,3) or actual_K.shape != (40,3,3) or not np.isfinite(upstream_K).all() or not np.isfinite(actual_K).all():
        raise ValueError('Frontend and exported intrinsics must be finite 40x3x3 arrays')
    if len(actual) != 40 or upstream.shape != actual.shape or not np.array_equal(ids, np.arange(40)) or not np.all(cameras == '0'):
        raise ValueError('Frontend/asset frame and camera identities do not match the 40-frame protocol')
    for poses in (upstream, actual):
        if not np.isfinite(poses).all() or not np.allclose(poses[:,3], [0,0,0,1], atol=1e-5):
            raise ValueError('Invalid frontend or exported homogeneous camera poses')
        rotations = poses[:,:3,:3]
        if not np.allclose(rotations.transpose(0,2,1) @ rotations, np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotations),1,atol=1e-4):
            raise ValueError('Camera transforms contain scale/shear instead of rigid rotations')
    transform = actual[0] @ np.linalg.inv(upstream[0])
    if not np.allclose(transform[:3,:3].T @ transform[:3,:3], np.eye(3), atol=1e-5) or abs(np.linalg.det(transform[:3,:3])-1) > 1e-5:
        raise ValueError('Frontend-to-asset conversion changed metric scale')
    predicted = transform[None] @ upstream
    translation_error = np.linalg.norm(predicted[:,:3,3]-actual[:,:3,3],axis=1)
    rotation_error = float(np.max(np.abs(predicted[:,:3,:3]-actual[:,:3,:3])))
    intrinsics_error = float(np.max(np.abs(upstream_K-actual_K)))
    if translation_error.max() > .005 or rotation_error > 1e-4 or intrinsics_error > .001:
        raise ValueError('Final asset is not the same metric camera sequence under a single rigid transform')
    return {'metres_per_native_unit': 1., 'source': 'verified MoGe/GCA frontend metric coordinate contract',
            'validation_scope': 'upstream_estimated_coordinate_units', 'absolute_accuracy_certified': False,
            'temporal_scale_checked': True, 'frame_count': 40,
            'max_camera_translation_error_m': float(translation_error.max()),
            'max_camera_rotation_matrix_error': rotation_error, 'max_intrinsics_error': intrinsics_error,
            'frontend_to_asset_rigid_transform': transform.tolist(),
            'frontend_metadata_path': str(metadata_path), 'frontend_metadata_sha256': digest(metadata_path),
            'frontend_poses_path': str(poses_path), 'frontend_poses_sha256': digest(poses_path),
            'note': 'One asset unit preserves one estimated frontend meter. This does not certify dense Gaussian geometry or real-world distance accuracy.'}
