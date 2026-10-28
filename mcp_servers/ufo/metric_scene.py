"""Persist and validate one metric scale and local road plane per reconstruction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .metric_config import number, vector


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruction_identity(workspace: Path, asset_root: Path) -> dict:
    asset = asset_root / 'assets' / workspace.name
    manifest = asset / 'ASSET.json'
    data = json.loads(manifest.read_text())
    if data.get('version') != 'ufo-worldengine-time-v2' or data.get('asset_id') != workspace.name:
        raise ValueError('Metric editing requires the original requested UFO time-v2 asset')
    sources = []
    for path in sorted((workspace / 'reconstruction/000/pred100k/sources').glob('*.npz')):
        stat = path.stat()
        sources.append([path.name, stat.st_size, stat.st_mtime_ns])
    if not sources:
        raise ValueError('Metric calibration requires original per-frame reconstruction sources')
    payload = {'workspace': str(workspace.resolve()), 'manifest_sha256': digest(manifest),
               'camera_sha256': digest(asset / 'cameras.npz'), 'source_stats': sources,
               'actor_stats': [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in sorted((asset / 'dynamic').glob('*.pt'))],
               'checkpoint_stat': [data['asset_ckpt'], Path(data['asset_ckpt']).stat().st_size, Path(data['asset_ckpt']).stat().st_mtime_ns],
               'build_sha256': digest(workspace / 'reconstruction/000/pred100k/BUILD.json')}
    return {'fingerprint': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), **payload}


def fit_road_plane(points, scale: float) -> dict:
    """Robust local planar road estimate in native world coordinates."""
    import numpy as np
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        raise ValueError('Road calibration needs >=100 finite static road points')
    rng = np.random.default_rng(20261022)
    threshold = .15 / scale
    best = None
    for _ in range(256):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        if abs(normal[2]) < .8:
            continue
        distance = -float(normal @ a)
        keep = np.abs(points @ normal + distance) < threshold
        if best is None or keep.sum() > best.sum():
            best = keep
    if best is None or best.mean() < .7:
        raise ValueError('Road plane is not coherent across frames; provide a measured local plane or repair alignment')
    center = np.median(points[best], axis=0)
    _, singular, vt = np.linalg.svd(points[best]-center, full_matrices=False)
    if singular[1] < 1e-5 or singular[1] / singular[0] < .01:
        raise ValueError('Road samples are degenerate/collinear')
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    distance = -float(normal @ center)
    errors = np.abs(points @ normal + distance) * scale
    keep = errors < .15
    if keep.mean() < .7 or normal[2] < .8:
        raise ValueError('Refined road plane failed coherence/orientation validation')
    return {'normal_world': normal.tolist(), 'offset_native': distance,
            'inlier_fraction': float(keep.mean()), 'inlier_p90_error_m': float(np.quantile(errors[keep], .9)),
            'sample_count': len(points), 'source': 'RANSAC static semantic road; local plane approximation'}


def road_samples(workspace: Path, manifest: dict, scale: float):
    import numpy as np
    rotation = np.asarray(manifest['source_to_world_rotation'], dtype=np.float64)
    origin = np.asarray(manifest['source_origin'], dtype=np.float64)
    rows = []
    for path in sorted((workspace / 'reconstruction/000/pred100k/sources').glob('*.npz')):
        with np.load(path, allow_pickle=False) as source:
            xyz = (source['means']-source['C'][:3, 3]) @ source['C'][:3, :3]
            keep = ((source['classes'] == 0) & ~source['dynamic'] & (source['coverage'] < .01)
                    & (source['opacities'] > .1) & np.isfinite(xyz).all(-1)
                    & (xyz[..., 2] * scale > 2) & (xyz[..., 2] * scale < 60))
            points = (source['means'][keep]-origin) @ rotation.T
            if len(points):
                rows.append(points[::max(1, (len(points)+999)//1000)])
    if not rows:
        raise ValueError('No usable static road samples; supply an explicit road-plane reference')
    return np.concatenate(rows)


def rendered_road_samples(workspace: Path, provider, indices, scale: float, depth_range_m=(2,60)):
    """Backproject the same visible expected-depth surface used for scale fitting."""
    import numpy as np
    rows = []
    for index in indices:
        with np.load(workspace / f'reconstruction/000/pred100k/sources/{index:06d}.npz', allow_pickle=False) as source:
            road = (source['classes'] == 0) & ~source['dynamic'] & (source['coverage'] < .01)
        depth, valid = provider.get(index, road.shape)
        keep = road & valid & (depth*scale > depth_range_m[0]) & (depth*scale < depth_range_m[1])
        y, x = np.where(keep)
        stride = max(1, (len(x)+999)//1000)
        y, x = y[::stride], x[::stride]
        rays = np.column_stack([x+.5, y+.5, np.ones(len(x))]) @ np.linalg.inv(provider.K[index]).T
        camera = rays*depth[y, x, None]
        pose = provider.poses[index]
        if len(x):
            rows.append(camera @ pose[:3, :3].T+pose[:3, 3])
    if not rows:
        raise ValueError('No visible rendered static road samples for the local road plane')
    return np.concatenate(rows)


def motion_ground_plane(poses, initial_plane: dict, near_points, scale: float) -> dict:
    """Local flat-driving approximation: travel tangent + near-road height.

    Straight vehicle motion constrains road pitch independently of a compressed
    far-depth surface. Roll comes from the rendered local patch. No surveyed
    gravity, camera height, or globally valid road is asserted.
    """
    import numpy as np
    centers=np.asarray(poses,dtype=np.float64)[:,:3,3]
    _,singular,vt=np.linalg.svd(centers-centers.mean(0),full_matrices=False)
    if np.linalg.norm(centers[-1]-centers[0])*scale < 5 or singular[1]/singular[0] > .01:
        raise ValueError('Motion-based road reference requires >=5m of approximately straight driving')
    tangent=vt[0]
    normal=np.asarray(initial_plane['normal_world'],dtype=np.float64).copy()
    normal-=tangent*float(normal@tangent)
    normal/=np.linalg.norm(normal)
    if normal[2]<0:normal=-normal
    if normal[2]<.8:
        raise ValueError('Motion-derived road normal is inconsistent with the world up convention')
    points=np.asarray(near_points,dtype=np.float64)
    if len(points)<100 or not np.isfinite(points).all():
        raise ValueError('Motion-based road height requires >=100 finite near-road points')
    offset=-float(np.median(points@normal))
    errors=np.abs(points@normal+offset)*scale
    keep=errors<.15
    if keep.mean()<.7:
        raise ValueError('Motion-derived local road height failed the unchanged 15cm/70% support gate')
    return {'normal_world':normal.tolist(),'offset_native':offset,
            'source':'straight metric camera trajectory normal + rendered near-road median height',
            'scope':'local_reference_patch','reference_frame':0,'global_road_surface_validated':False,
            'inlier_fraction':float(keep.mean()),'inlier_p90_error_m':float(np.quantile(errors[keep],.9)),
            'sample_count':len(points),'camera_motion_secondary_ratio':float(singular[1]/singular[0]),
            'assumptions':['Approximately straight driving on a locally planar road; not surveyed gravity or height.']}


def calibrate(workspace: Path, asset_root: Path, output: Path, *, reference: Path | None = None,
              road_plane: Path | None = None, moge_source: Path | None = None,
              moge_checkpoint: Path | None = None, force=False, method='depth_fit', road_model='local_depth') -> dict:
    from .service import DEFAULT_MOGE_ROOT, DEFAULT_MOGE_CHECKPOINT
    import os
    workspace, asset_root, output = workspace.resolve(), asset_root.resolve(), output.resolve()
    protected = [asset_root / 'assets', workspace / 'reconstruction', workspace / 'input']
    if output.suffix.lower() != '.json' or any(output.is_relative_to(root.resolve()) for root in protected):
        raise ValueError('Calibration output must be a separate .json outside source assets/reconstruction/input')
    if any(path and output == path.resolve() for path in (reference, road_plane)):
        raise ValueError('Calibration output must not overwrite its reference inputs')
    identity = reconstruction_identity(workspace, asset_root)
    if method not in {'depth_fit', 'frontend_metric'} or (reference and method != 'depth_fit'):
        raise ValueError('Use method=depth_fit or frontend_metric; measured references use depth_fit')
    if road_model not in {'local_depth','motion_near_depth'} or (road_model=='motion_near_depth' and method!='frontend_metric'):
        raise ValueError('motion_near_depth road_model requires the verified frontend_metric method')
    if road_plane and road_model!='local_depth':
        raise ValueError('Choose either an explicit road-plane file or a motion-derived road model')
    frontend_fit = None
    if method == 'frontend_metric':
        from .frontend_units import verify_frontend_units
        frontend_fit = verify_frontend_units(workspace, asset_root)
    teacher_needed = not reference and method == 'depth_fit'
    source = (moge_source or Path(os.environ.get('UFO_MOGE_SOURCE', str(DEFAULT_MOGE_ROOT / 'vendor_moge')))).resolve()
    checkpoint = (moge_checkpoint or Path(os.environ.get('UFO_MOGE_CHECKPOINT', str(DEFAULT_MOGE_CHECKPOINT)))).resolve()
    request = {'version': 3, 'method': method, 'road_model':road_model, 'native_depth_definition': 'rendered_expected_camera_z_v1',
               'native_depth_helper_sha256': digest(Path(__file__).with_name('rendered_depth.py')) if not reference else None,
               'frontend_provenance': ({key: frontend_fit[key] for key in
                   ('source','validation_scope','frontend_metadata_sha256','frontend_poses_sha256')} if frontend_fit else None),
               'reference_sha256': digest(reference) if reference else None,
               'road_plane_sha256': digest(road_plane) if road_plane else None,
               'moge_source': str(source) if teacher_needed else None,
               'moge_model_sha256': digest(source / 'moge/model/v2.py') if teacher_needed else None,
               'checkpoint': [str(checkpoint), checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns] if teacher_needed else None}
    if output.is_file() and not force:
        cached = load_calibration(output, workspace, asset_root)
        if cached.get('request') != request:
            raise ValueError('Calibration inputs changed; use a distinct output path or --force')
        print(f'RESUME metric calibration: {output}', flush=True)
        return cached
    if frontend_fit is not None:
        fit, scale = frontend_fit, 1.
        # Keep any independent failed depth agreement visible; this path never
        # converts a failed geometric fit into a successful depth certification.
        diagnostics = output.parent / 'moge_metric_depths' / identity['fingerprint'][:16] / 'rendered_depth_diagnostics'
        fit['independent_depth_reports'] = [str(p) for p in sorted(diagnostics.glob('frame_*.json'))]
    elif reference:
        value = json.loads(reference.read_text())
        if value.get('source') != 'measured_reference' or not isinstance(value.get('description'), str) or not value['description'].strip():
            raise ValueError('Reference must identify source=measured_reference and describe the real measurement')
        scale = number(value.get('distance_m'), 'distance_m', positive=True) / number(value.get('native_distance'), 'native_distance', positive=True)
        uncertainty = number(value.get('relative_uncertainty'), 'relative_uncertainty')
        if not 0 <= uncertainty < 1:
            raise ValueError('Reference relative_uncertainty must be in [0,1)')
        fit = {'metres_per_native_unit': scale, 'source': 'measured_reference',
               'relative_uncertainty': uncertainty, 'reference': value,
               'temporal_scale_checked': False}
    else:
        from .metric_edit import calibrate_workspace
        fit = calibrate_workspace(workspace, moge_source=source, moge_checkpoint=checkpoint,
                                  depth_cache=output.parent / 'moge_metric_depths' / identity['fingerprint'][:16],
                                  camera_file=asset_root / 'assets' / workspace.name / 'cameras.npz', asset_root=asset_root)
        scale = fit['metres_per_native_unit']
        fit['temporal_scale_checked'] = True
    manifest = json.loads((asset_root / 'assets' / workspace.name / 'ASSET.json').read_text())
    if road_plane:
        plane = json.loads(road_plane.read_text())
        normal = vector(plane.get('normal_world'), 'normal_world')
        if abs(sum(v*v for v in normal)-1) > 1e-5 or normal[2] < .8:
            raise ValueError('Road normal must be unit length and point up in UFO world coordinates')
        plane = {'normal_world': normal, 'offset_native': number(plane.get('offset_native'), 'offset_native'),
                 'source': 'user supplied native world plane', 'description': plane.get('description', '')}
    else:
        if reference:
            points = road_samples(workspace, manifest, scale)
        else:
            from .rendered_depth import RenderedDepthCache
            provider = RenderedDepthCache(workspace, asset_root,
                output.parent / 'moge_metric_depths' / identity['fingerprint'][:16] / 'rendered_native_v1')
            try:
                if method == 'frontend_metric':
                    # Coordinate units do not require one globally planar road
                    # over the entire clip. Establish an explicitly LOCAL patch
                    # for the editing reference frame, retaining the same gate.
                    indices = [0]
                else:
                    indices = [row['frame'] for row in fit['frame_fits']]
                points = rendered_road_samples(workspace, provider, indices, scale)
                near_points = (rendered_road_samples(workspace,provider,[0],scale,(2,8))
                               if road_model=='motion_near_depth' else None)
            finally:
                provider.close()
        plane = fit_road_plane(points, scale)
        if not reference:
            plane['source'] = 'RANSAC static road backprojected from rendered expected camera_z'
        if method == 'frontend_metric':
            plane.update(scope='local_reference_patch', reference_frame=0,
                         global_road_surface_validated=False)
        if road_model=='motion_near_depth':
            plane=motion_ground_plane(provider.poses,plane,near_points,scale)
    result = {'schema_version': 1, 'kind': 'ufo_metric_scene', 'status': 'validated',
              'reconstruction': identity, 'request': request, 'scale_fit': fit,
              'meters_per_native_unit': scale, 'road_plane': plane,
              'validation_scope': fit.get('validation_scope', 'depth_agreement' if not reference else 'measured_scale_reference'),
              'accuracy_note': 'Execution precision is distinct from real-world scale/road estimation accuracy; no independent physical accuracy certification.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(output)
    return result


def load_calibration(path: Path, workspace: Path, asset_root: Path) -> dict:
    import numpy as np
    value = json.loads(path.read_text())
    if value.get('kind') != 'ufo_metric_scene' or value.get('status') != 'validated':
        raise ValueError('Not a validated UFO metric calibration')
    if value.get('reconstruction', {}).get('fingerprint') != reconstruction_identity(workspace, asset_root)['fingerprint']:
        raise ValueError('Metric calibration is stale or belongs to another reconstruction')
    number(value.get('meters_per_native_unit'), 'meters_per_native_unit', positive=True)
    if value.get('validation_scope') == 'upstream_estimated_coordinate_units':
        from .frontend_units import verify_frontend_units
        current = verify_frontend_units(workspace, asset_root)
        for key in ('frontend_metadata_sha256','frontend_poses_sha256'):
            if current[key] != value['scale_fit'].get(key):
                raise ValueError('Upstream metric-unit provenance changed after calibration')
    normal = np.asarray(vector(value['road_plane'].get('normal_world'), 'normal_world'))
    if abs(float(normal @ normal)-1) > 1e-5 or normal[2] < .8:
        raise ValueError('Invalid calibrated road normal')
    number(value['road_plane'].get('offset_native'), 'offset_native')
    return value


def road_frame(pose, plane: dict):
    """Fixed FLU basis and road-projected camera origin, not a surveyed ego axle."""
    import numpy as np
    up = np.asarray(plane['normal_world'], dtype=np.float64)
    forward = np.asarray(pose[:3, 2], dtype=np.float64).copy()
    forward -= up * float(forward @ up)
    if np.linalg.norm(forward) < .1:
        raise ValueError('Cannot infer a horizontal forward direction from this camera')
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    origin = np.asarray(pose[:3, 3], dtype=np.float64).copy()
    origin -= up * (float(origin @ up)+plane['offset_native'])
    return np.column_stack((forward, left, up)), origin
