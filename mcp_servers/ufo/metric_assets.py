"""Known-unit Gaussian assets, converted to one rigid local FLU object."""
from __future__ import annotations

import json
from pathlib import Path

from .metric_config import number, vector
from .metric_scene import digest

FIELDS = ('means', 'scales', 'quats', 'opacities', 'colors')


def rotation_quaternion(matrix):
    """Proper rotation to unit wxyz quaternion, without an extra geometry library."""
    import numpy as np
    m = np.asarray(matrix, dtype=np.float64)
    if np.trace(m) > 0:
        s = np.sqrt(1+np.trace(m))*2
        q = np.array([s/4, (m[2, 1]-m[1, 2])/s, (m[0, 2]-m[2, 0])/s, (m[1, 0]-m[0, 1])/s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i+1) % 3, (i+2) % 3
        s = np.sqrt(1+m[i, i]-m[j, j]-m[k, k])*2
        q = np.zeros(4)
        q[0], q[i+1], q[j+1], q[k+1] = (m[k, j]-m[j, k])/s, s/4, (m[j, i]+m[i, j])/s, (m[k, i]+m[i, k])/s
    return q / np.linalg.norm(q)


def rotate_quaternions(quaternions, rotation):
    import numpy as np
    q = rotation_quaternion(rotation)
    values = np.asarray(quaternions, dtype=np.float64)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    w, xyz = values[:, :1], values[:, 1:]
    result = np.concatenate([q[0]*w-(xyz*q[1:]).sum(-1, keepdims=True),
                             q[0]*xyz+w*q[1:]+np.cross(q[1:], xyz)], axis=1)
    return result.astype(np.float32)


def validate_gaussians(data: dict) -> dict:
    import numpy as np
    missing = set(FIELDS)-set(data)
    if missing:
        raise ValueError(f'Gaussian asset missing fields: {sorted(missing)}')
    arrays = {}
    for key in FIELDS:
        value = data[key]
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        arrays[key] = np.asarray(value, dtype=np.float32)
    n = len(arrays['means'])
    arrays['opacities'] = arrays['opacities'].reshape(-1)
    if not n or any(arrays[k].shape != shape for k, shape in {
        'means': (n, 3), 'scales': (n, 3), 'quats': (n, 4), 'colors': (n, 3), 'opacities': (n,)}.items()):
        raise ValueError('Invalid Gaussian shapes; expected Nx3 means/scales/colors, Nx4 wxyz quats, N opacities')
    if any(not np.isfinite(a).all() for a in arrays.values()) or (arrays['scales'] <= 0).any():
        raise ValueError('Gaussian fields must be finite; scales must be positive physical lengths')
    if ((arrays['opacities'] < 0) | (arrays['opacities'] > 1)).any():
        raise ValueError('Opacities must be activated probabilities, not logits')
    lengths = np.linalg.norm(arrays['quats'], axis=1)
    if (lengths < 1e-8).any():
        raise ValueError('Zero Gaussian quaternion')
    arrays['quats'] = arrays['quats']/lengths[:, None]
    return arrays


def load_asset(path: Path) -> tuple[dict, dict]:
    import numpy as np
    path = path.resolve()
    manifest = json.loads(path.read_text())
    asset = Path(manifest['asset_path']).expanduser()
    if not asset.is_absolute():
        asset = path.parent / asset
    asset = asset.resolve()
    if manifest.get('color_space') not in {'linear_rgb', 'srgb'}:
        raise ValueError('Asset must declare color_space=linear_rgb or srgb')
    if manifest.get('quaternion_order') != 'wxyz':
        raise ValueError('Asset must declare quaternion_order=wxyz')
    rotation = np.asarray(manifest.get('asset_to_flu'), dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < .999:
        raise ValueError('asset_to_flu must be a proper orthonormal 3x3 rotation')
    pivot = np.asarray(vector(manifest.get('ground_contact_point'), 'ground_contact_point'))
    kind = manifest.get('format')
    appearance = None
    if kind == 'gaussian_npz':
        with np.load(asset, allow_pickle=False) as data:
            arrays = validate_gaussians(data)
    elif kind == 'gaussian_pt':
        import torch
        arrays = validate_gaussians(torch.load(asset, map_location='cpu', weights_only=True))
    elif kind == '3dgs_ply':
        if manifest.get('appearance') != 'dc_only':
            raise ValueError('3DGS PLY import requires explicit appearance=dc_only (view-dependent SH is not imported)')
        from plyfile import PlyData
        vertices = PlyData.read(str(asset))['vertex'].data
        required = ['x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
                    'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2']
        if not set(required) <= set(vertices.dtype.names):
            raise ValueError('PLY must contain 3DGS scales, wxyz rotations, opacity logits and SH DC; point-only PLY is unsupported')
        def stack(names):
            return np.column_stack([vertices[name] for name in names]).astype(np.float64)
        logits = np.clip(vertices['opacity'].astype(np.float64), -80, 80)
        arrays = validate_gaussians({'means': stack(['x', 'y', 'z']),
            'scales': np.exp(stack([f'scale_{i}' for i in range(3)])),
            'quats': stack([f'rot_{i}' for i in range(4)]), 'opacities': 1/(1+np.exp(-logits)),
            'colors': .5 + .28209479177387814*stack([f'f_dc_{i}' for i in range(3)])})
        appearance = 'DC-only appearance; higher-order view-dependent SH discarded by explicit manifest policy'
    else:
        raise ValueError('Supported asset formats: gaussian_npz, gaussian_pt, 3dgs_ply')
    local = (arrays['means'].astype(np.float64)-pivot) @ rotation.T
    dimensions = np.ptp(local, axis=0)
    if (dimensions <= 1e-8).any():
        raise ValueError('Asset must have a non-degenerate 3D extent')
    expected = manifest.get('dimensions_m')
    if expected is not None:
        expected = np.asarray(vector(expected, 'dimensions_m'))
        if (expected <= 0).any():
            raise ValueError('Asset dimensions_m must be positive [length,width,height]')
    unit = manifest.get('meters_per_unit')
    if unit is None:
        if expected is None:
            raise ValueError('Asset needs known meters_per_unit or dimensions_m; scale cannot be guessed')
        unit = float(np.median(expected/dimensions))
    unit = number(unit, 'asset meters_per_unit', positive=True)
    if expected is not None and np.max(np.abs(dimensions*unit-expected)/expected) > .10:
        raise ValueError('Asset physical dimensions disagree by >10%; no nonuniform resizing is applied')
    if abs(float(local[:, 2].min())*unit) > .15:
        raise ValueError('Declared ground_contact_point is not at the lower Gaussian mean surface (tolerance 0.15 m)')
    arrays['means'] = (local*unit).astype(np.float32)
    arrays['scales'] = arrays['scales']*unit
    arrays['quats'] = rotate_quaternions(arrays['quats'], rotation)
    if manifest['color_space'] == 'srgb':
        colors = arrays['colors']
        if ((colors < 0) | (colors > 1)).any():
            raise ValueError('sRGB asset colors must be in [0,1]')
        arrays['colors'] = np.where(colors <= .04045, colors/12.92, ((colors+.055)/1.055)**2.4).astype(np.float32)
    metadata = {'manifest_path': str(path), 'manifest_sha256': digest(path), 'asset_path': str(asset),
                'asset_sha256': digest(asset), 'meters_per_asset_unit': unit,
                'dimensions_m': (dimensions*unit).tolist(), 'gaussian_count': len(local),
                'appearance_limitation': appearance, 'grounding_note': 'Mean-surface contact; Gaussian support may extend below road'}
    metadata['dimension_source'] = manifest.get('dimension_source', 'declared asset units/dimensions')
    return arrays, metadata


def place_asset(local: dict, basis, origin_native, position_m, yaw_degrees, scale):
    import numpy as np
    theta = np.deg2rad(yaw_degrees)
    c, s = np.cos(theta), np.sin(theta)
    rotation = basis @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    result = {k: v.copy() for k, v in local.items()}
    result['means'] = (local['means'] @ rotation.T/scale + origin_native + basis @ np.asarray(position_m)/scale).astype(np.float32)
    result['scales'] = (local['scales']/scale).astype(np.float32)
    result['quats'] = rotate_quaternions(local['quats'], rotation)
    return result
