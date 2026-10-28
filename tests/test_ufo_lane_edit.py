import json
import tempfile
import unittest
import os
import subprocess
import sys
from pathlib import Path

from mcp_servers.ufo.lane_edit import lane_progress, choose_target

try:
    import numpy as np
    import torch
    from PIL import Image
except ImportError:
    np = torch = None


class LanePlanTests(unittest.TestCase):
    def test_offline_workers_block_external_connections(self):
        bootstrap = Path(__file__).resolve().parents[1] / 'mcp_servers/ufo/offline_bootstrap'
        code = '''import socket
s = socket.socket()
s.settimeout(.1)
try:
    s.connect(('203.0.113.1', 443))
except OSError as error:
    assert 'UFO offline mode blocked' in str(error), str(error)
else:
    raise AssertionError('external connection was allowed')
finally:
    s.close()
'''
        env = dict(os.environ, UFO_OFFLINE='1', PYTHONPATH=str(bootstrap), PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-c', code], env=env, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_smooth_progress_clamps_and_reaches_target(self):
        self.assertEqual([lane_progress(t, 1, 3) for t in (0, 1, 2, 3, 4)], [0, 0, .5, 1, 1])
        self.assertLess(lane_progress(1.01, 1, 3), .001)
        with self.assertRaises(ValueError):
            lane_progress(1, 3, 1)

    def test_sparse_target_is_not_automatically_used(self):
        candidates = [{'track_id': 1, 'visible_frames': 5}]
        with self.assertRaisesRegex(ValueError, '80%'):
            choose_target(candidates, None, 40)
        self.assertEqual(choose_target(candidates, 1, 40)['track_id'], 1)


@unittest.skipIf(np is None or torch is None, 'CPU Gaussian fixture needs numpy, torch and Pillow')
class LaneAssetTests(unittest.TestCase):
    def fixture(self, root):
        workspace = root / 'scene'
        source_dir = workspace / 'reconstruction/000/pred100k/sources'
        source_dir.mkdir(parents=True)
        reference = source_dir.parent / 'reference'
        reference.mkdir()
        original = root / 'original/assets/scene'
        (original / 'background').mkdir(parents=True)
        (original / 'dynamic').mkdir()
        rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
        camera = np.eye(4, dtype=np.float32)
        packaged = camera.copy()
        packaged[:3, :3] = rotation
        height, width = 8, 10
        yy, xx = np.mgrid[:height, :width]
        tracks = np.where(xx < 5, 10001, 10002).astype(np.int32)
        means = np.stack([(xx - 4.5) * .2, (yy - 3.5) * .1, np.full_like(xx, 10)], -1).astype(np.float32)
        paths = []
        for i in range(40):
            frame_means = means.copy()
            frame_means[:, :5, 0] += .01 * i  # Original target drifts laterally.
            np.savez(source_dir / f'{i:06d}.npz', means=frame_means, scales=np.ones_like(means),
                     C=camera, opacities=np.ones((height, width)), dynamic=np.ones((height, width), dtype=bool),
                     coverage=np.zeros((height, width)), tracks=tracks, classes=np.full((height, width), 13))
            Image.new('RGB', (520, 352), 'gray').save(reference / f'{i:06d}.png')
            actor = {'means': torch.from_numpy(frame_means.reshape(-1, 3) @ rotation.T),
                     'opacities': torch.ones(height * width), 'colors': torch.zeros(height * width, 3)}
            path = original / 'dynamic' / f'{i:06d}.pt'
            torch.save(actor, path)
            paths.append(str(path))
        ckpt = original / 'background/scene.ckpt'
        torch.save({'background': {'config': {}, 'state_dict': {}},
                    'observed_dynamic': {'config': {'actor_files': paths}}}, ckpt)
        K = np.array([[100, 0, 260], [0, 100, 176], [0, 0, 1]], dtype=np.float32)
        np.savez(original / 'cameras.npz', C=np.stack([packaged] * 40), K=np.stack([K] * 40), timestamps=np.arange(40) / 10)
        manifest = {'asset_id': 'scene', 'version': 'ufo-worldengine-time-v2', 'asset_ckpt': str(ckpt),
                    'source_to_world_rotation': rotation.tolist(), 'source_origin': [0, 0, 0]}
        (original / 'ASSET.json').write_text(json.dumps(manifest))
        return workspace, original, tracks

    def test_lane_edit_preserves_other_vehicle_height_and_source(self):
        from mcp_servers.ufo.lane_edit import edit
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            workspace, original, tracks = self.fixture(root)
            untouched = (original / 'dynamic/000039.pt').read_bytes()
            report = edit(workspace, root / 'original', root / 'edited', 'scene_right',
                          track_id=10001, direction='right', lateral_distance=2.)
            self.assertEqual(report['units'], 'native')
            self.assertFalse(report['metric_calibrated'])
            self.assertEqual((original / 'dynamic/000039.pt').read_bytes(), untouched)
            target = (tracks == 10001).reshape(-1)
            final_laterals = []
            for index in (0, 36, 39):
                before = torch.load(original / 'dynamic' / f'{index:06d}.pt', weights_only=False)
                after = torch.load(root / 'edited/assets/scene_right/dynamic' / f'{index:06d}.pt', weights_only=False)
                self.assertTrue(torch.equal(before['means'][~target], after['means'][~target]))
                self.assertTrue(torch.equal(before['means'][:, 2], after['means'][:, 2]))
                self.assertTrue(torch.equal(before['opacities'], after['opacities']))
                if index == 0:
                    self.assertTrue(torch.equal(before['means'], after['means']))
                else:
                    final_laterals.append(float((after['means'][target, 1] - before['means'][target, 1]).mean()))
            self.assertAlmostEqual(final_laterals[0], final_laterals[1], places=5)
            self.assertLess(report['frames'][-1]['requested_left_offset_native'], 0)
            self.assertTrue(Path(report['target_preview']).is_file())

    def test_visual_edit_preserves_silhouette_across_depth_outliers(self):
        from mcp_servers.ufo.lane_edit import view_consistent_delta
        points = np.array([[-1., 0, 5], [0., 0, 10], [4., 0, 40]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        K = np.array([[200, 0, 260], [0, 200, 176], [0, 0, 1]], dtype=np.float32)
        for depth_scale in [1., 2., .5]:
            before = points * depth_scale
            delta, error = view_consistent_delta(before, pose, K, np.array([-1, 0, 0]), 40.)
            after = before + delta
            uv_before = before @ K.T
            uv_after = after @ K.T
            self.assertTrue(np.allclose(uv_after[:, 0] / uv_after[:, 2] - uv_before[:, 0] / uv_before[:, 2], 40))
            self.assertTrue(np.array_equal(after[:, 1:], before[:, 1:]))
            self.assertLess(error, .001)

    def test_visual_asset_keeps_other_actors_and_height(self):
        from mcp_servers.ufo.lane_edit import edit
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            workspace, original, tracks = self.fixture(root)
            report = edit(workspace, root / 'original', root / 'visual', 'visual_edit', track_id=10001)
            self.assertEqual(report['profile'], 'view_consistent_v2')
            target = (tracks == 10001).reshape(-1)
            before = torch.load(original / 'dynamic/000039.pt', weights_only=False)
            after = torch.load(root / 'visual/assets/visual_edit/dynamic/000039.pt', weights_only=False)
            self.assertTrue(torch.equal(after['means'][~target], before['means'][~target]))
            self.assertTrue(torch.equal(after['means'][:, 2], before['means'][:, 2]))
            self.assertEqual(report['frames'][0]['image_offset_pixels'], 0)
            self.assertGreater(report['frames'][-1]['image_offset_pixels'], 0)
            self.assertLess(max(row['projection_error_pixels'] for row in report['frames']), .02)

    def test_native_candidates_keep_all_tracks_and_selector_schema(self):
        from mcp_servers.ufo.candidates import discover
        from tools.tool05_instance_discovery.evidence import original_box
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            workspace, original, tracks = self.fixture(root)
            rgb = workspace / 'input/rgb/video/0'
            rgb.mkdir(parents=True)
            for i in range(40):
                Image.new('RGB', (1040, 704), 'gray').save(rgb / f'{i:06d}.jpg')
            result = discover(workspace, root / 'candidate/instances.json')
            self.assertTrue(result['selection_required'])
            self.assertEqual([i['sam_track_id'] for i in result['instances']], [10001, 10002])
            self.assertEqual(len(result['object_cards']), 2)
            self.assertEqual(len(result['frame_images']), 40)
            box = original_box(result['instances'][0]['frames'][0]['bbox'], (1040, 704))
            self.assertLess(abs(box[2] - 520), 3)


if __name__ == '__main__':
    unittest.main()
