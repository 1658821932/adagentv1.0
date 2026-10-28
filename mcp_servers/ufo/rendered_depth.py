"""Expected camera-z of the final UFO asset, using the native Gaussian renderer.

Gaussian-center camera-z is NOT the depth image supervised by UFO's RGB+ED
training loss. This cache renders visibility/opacity/covariance composition.
No UFO/MoGe forward pass, sky network or neural RGB background is required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


class RenderedDepthCache:
    def __init__(self, workspace: Path, asset_root: Path, directory: Path,
                 *, worldengine_root: Path | None = None):
        import numpy as np
        from .runtime import resources
        from .metric_scene import reconstruction_identity, digest
        self.asset_root, self.asset_id = asset_root.resolve(), workspace.name
        self.directory = directory
        self.worldengine = (worldengine_root or resources()['worldengine']).resolve()
        with np.load(self.asset_root / 'assets' / self.asset_id / 'cameras.npz', allow_pickle=False) as camera:
            self.poses, self.K, self.times, self.ego = [camera[key].copy() for key in ('C', 'K', 'timestamps', 'ego')]
        simengine = self.worldengine / 'projects/SimEngine'
        code = [simengine / 'worldengine/render/mtgs/mtgs.py',
                simengine / 'worldengine/render/mtgs/gaussian_model/ufo_portable.py']
        contract = {'definition': 'rendered_expected_camera_z_v1', 'reconstruction': reconstruction_identity(workspace, self.asset_root)['fingerprint'],
                    'renderer': {str(p): digest(p) for p in code}, 'near': .2, 'far': 400.,
                    'rasterize_mode': 'classic', 'render_mode': 'RGB+ED', 'helper_sha256': digest(Path(__file__))}
        self.key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        self.engine = None

    @staticmethod
    def decode(depth, alpha, shape):
        import numpy as np
        depth, alpha = np.asarray(depth), np.asarray(alpha)
        if depth.shape != shape or alpha.shape != shape:
            raise ValueError('Rendered depth/alpha must match the reference pixel grid')
        valid = np.isfinite(depth) & np.isfinite(alpha) & (alpha >= .5) & (depth > .2) & (depth < 400)
        return depth.astype(np.float32), valid

    def get(self, frame: int, shape):
        import numpy as np
        path = self.directory / f'{frame:06d}_{self.key}.npz'
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                depth, valid = self.decode(data['depth'], data['alpha'], shape)
            print(f'RESUME rendered native depth frame={frame}', flush=True)
            return depth, valid
        print(f'RENDER native expected depth frame={frame}', flush=True)
        depth, alpha = self._render(frame, shape)
        depth, valid = self.decode(depth, alpha, shape)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, depth=depth, alpha=alpha)
        temporary.replace(path)
        return depth, valid

    def _render(self, frame: int, shape):
        import numpy as np
        import torch
        sys.path[:0] = [str(self.worldengine / 'vendor_runtime'), str(self.worldengine / 'projects/SimEngine')]
        from worldengine.render.mtgs.mtgs import MTGSRenderEngine, rasterization
        if self.engine is None:
            self.engine = MTGSRenderEngine(device='cuda', asset_folder_path=self.asset_root / 'assets',
                                          rasterize_mode='classic', near_plane=.2, far_plane=400., packed=True)
            self.engine.reset(self.asset_id, {'metadata': {'digitaltwin_asset_id': self.asset_id,
                                                          'digitaltwin_ego2globals': list(self.ego)}})
        height, width = shape
        with torch.inference_mode():
            self.engine.update_world(round(float(self.times[frame])*1e6), {})
            gaussians = self.engine.collected_gaussians
            rendered, alpha, _ = rasterization(
                **{key: gaussians[key] for key in ('means', 'quats', 'scales', 'opacities')},
                colors=torch.zeros((len(gaussians['means']), 3), device='cuda', dtype=torch.float32),
                viewmats=torch.as_tensor(np.linalg.inv(self.poses[frame])[None], device='cuda', dtype=torch.float32),
                Ks=torch.as_tensor(self.K[frame][None], device='cuda', dtype=torch.float32),
                width=width, height=height, tile_size=16, packed=True, near_plane=.2, far_plane=400.,
                render_mode='RGB+ED', sparse_grad=False, absgrad=True, rasterize_mode='classic', radius_clip=0.)
            return rendered[0, ..., -1].float().cpu().numpy(), alpha[0, ..., 0].float().cpu().numpy()

    def close(self):
        if self.engine is not None:
            import torch
            self.engine = None
            torch.cuda.empty_cache()
