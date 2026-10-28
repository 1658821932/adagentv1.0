import contextlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

try:
    import numpy as np
    import torch
    from PIL import Image
except ImportError:
    np = torch = None


@unittest.skipIf(np is None, 'Depth tests need numpy, torch and Pillow')
class RenderedDepthTests(unittest.TestCase):
    def test_road_points_backproject_rendered_surface_and_ignore_center_geometry(self):
        from mcp_servers.ufo.metric_scene import rendered_road_samples
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root/'reconstruction/000/pred100k/sources'
            source.mkdir(parents=True)
            np.savez(source/'000000.npz', classes=np.zeros((4,5),int), dynamic=np.zeros((4,5),bool),
                     coverage=np.zeros((4,5)), means=np.ones((4,5,3))*999)
            provider = Mock()
            provider.K = np.stack([np.eye(3)])
            provider.poses = np.stack([np.eye(4)])
            provider.get.return_value = (np.full((4,5),10.), np.ones((4,5),bool))
            points = rendered_road_samples(root, provider, [0], 1.)
            self.assertTrue(np.allclose(points[:,2],10))
            self.assertEqual(len(points),20)

    def test_reuses_rendered_depth_without_loading_gpu_engine(self):
        from mcp_servers.ufo.rendered_depth import RenderedDepthCache
        with tempfile.TemporaryDirectory() as directory:
            provider = RenderedDepthCache.__new__(RenderedDepthCache)
            provider.directory, provider.key, provider.engine = Path(directory), 'test', None
            expected = np.full((4, 5), 12., dtype=np.float32)
            alpha = np.ones((4, 5), dtype=np.float32)
            alpha[0, 0] = .1
            with patch.object(provider, '_render', return_value=(expected, alpha)) as render:
                depth, valid = provider.get(0, expected.shape)
                self.assertFalse(valid[0, 0])
                self.assertEqual(valid.sum(), 19)
                repeat, repeat_valid = provider.get(0, expected.shape)
                render.assert_called_once()
                self.assertTrue(np.array_equal(depth, repeat))
                self.assertTrue(np.array_equal(valid, repeat_valid))

    def test_calibration_uses_rendered_expected_depth_instead_of_gaussian_centers(self):
        from mcp_servers.ufo.metric_edit import calibrate_workspace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            scene = root/'reconstruction/000/pred100k'
            (scene/'sources').mkdir(parents=True)
            (scene/'reference').mkdir()
            (scene/'BUILD.json').write_text(json.dumps({'all_sources': 40}))
            model_file = root/'teacher/moge/model/v2.py'
            model_file.parent.mkdir(parents=True)
            model_file.write_text('# synthetic model source')
            checkpoint = root/'model.pt'
            checkpoint.write_bytes(b'not loaded when teacher depths are cached')
            yy, xx = np.mgrid[:64, :64]
            centers = np.stack([xx*0., yy*0., 1+xx*.7], -1).astype(np.float32)
            # Intentionally very different Gaussian-center z values; they are
            # not the expected per-pixel depth of an overlapping rendered scene.
            for frame in (0, 10, 20, 29, 39):
                np.savez(scene/'sources'/f'{frame:06d}.npz', means=centers, C=np.eye(4),
                         dynamic=np.zeros((64,64), bool), coverage=np.zeros((64,64)),
                         opacities=np.ones((64,64))*.1, classes=np.zeros((64,64), int))
                Image.new('RGB', (64,64)).save(scene/'reference'/f'{frame:06d}.png')
            K = np.array([[50.,0,32],[0,50,32],[0,0,1]])
            camera_file = root/'cameras.npz'
            np.savez(camera_file, K=np.stack([K]*40))
            cache = root/'teacher_cache'
            expected = np.full((64,64), 10., dtype=np.float32)
            provider = Mock()
            provider.get.return_value = (expected, np.ones((64,64), bool))
            teacher = types.ModuleType('moge.model.v2')
            teacher.MoGeModel = Mock(side_effect=AssertionError('Cached calibration must not load a teacher model'))
            real_load, real_is_file = np.load, Path.is_file

            def is_file(path):
                return path.parent == cache or real_is_file(path)

            def load(path, *args, **kwargs):
                if Path(path).parent == cache:
                    return contextlib.nullcontext({'depth': expected*2, 'mask': np.ones((64,64),bool)})
                return real_load(path, *args, **kwargs)

            with patch.dict('sys.modules', {'moge.model.v2': teacher}), \
                 patch('mcp_servers.ufo.rendered_depth.RenderedDepthCache', return_value=provider), \
                 patch.object(Path, 'is_file', is_file), patch('numpy.load', side_effect=load):
                result = calibrate_workspace(root, moge_source=root/'teacher', moge_checkpoint=checkpoint,
                    camera_file=camera_file, asset_root=root/'assets_root', depth_cache=cache)
            self.assertAlmostEqual(result['metres_per_native_unit'], 2.)
            self.assertEqual(result['native_depth_definition'], 'rendered_expected_camera_z_v1')
            self.assertEqual(provider.get.call_count, 5)
            teacher.MoGeModel.assert_not_called()
            provider.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
