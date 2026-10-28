import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from mcp_servers.ufo.model_loading import install_camera_count_adapter
from mcp_servers.ufo.resume import check_resume_contract, stage_complete


class CameraLoaderTests(unittest.TestCase):
    def test_uses_single_camera_without_modifying_original_weight_object(self):
        inference = SimpleNamespace()
        weight = object()
        def factory(*, num_cams=3, **kwargs):
            return {'num_cams': num_cams, 'weight': weight}
        registry = {'UFO': factory}
        inference.UFO_models = registry
        def original(args, device):
            return inference.UFO_models[args.model](args=args)
        inference.build_model = original
        args = SimpleNamespace(model='UFO', num_max_cameras=1)
        self.assertEqual(original(args, 'cpu')['num_cams'], 3)
        install_camera_count_adapter(inference)
        model = inference.build_model(args, 'cpu')
        self.assertEqual(model['num_cams'], 1)
        self.assertIs(model['weight'], weight)
        self.assertIs(inference.UFO_models, registry)

    def test_loader_errors_are_not_suppressed_and_registry_is_restored(self):
        inference = SimpleNamespace(UFO_models={'UFO': lambda **kwargs: None})
        registry = inference.UFO_models
        def original(args, device):
            raise RuntimeError('checkpoint mismatch')
        inference.build_model = original
        install_camera_count_adapter(inference)
        with self.assertRaisesRegex(RuntimeError, 'checkpoint mismatch'):
            inference.build_model(SimpleNamespace(model='UFO', num_max_cameras=1), 'cpu')
        self.assertIs(inference.UFO_models, registry)


class ResumeTests(unittest.TestCase):
    def contract(self):
        return {'video': '/video.mp4', 'sha256': 'video-hash', 'frames': 40, 'python': '/ufoex/python',
                'checkpoint': {'path': '/model.pth', 'size': 123}, 'source': '/model/source',
                'code': {'/test/mcp_servers/ufo/stage_runner.py': 'before', '/model/source/ufo/model.py': 'same'}}

    def test_explicit_resume_accepts_only_adapter_updates(self):
        previous = self.contract()
        current = copy.deepcopy(previous)
        current['code']['/test/mcp_servers/ufo/stage_runner.py'] = 'after'
        current['code']['/test/mcp_servers/ufo/model_loading.py'] = 'added'
        with self.assertRaisesRegex(ValueError, '--resume'):
            check_resume_contract(previous, current, allow_adapter_updates=False)
        self.assertEqual(len(check_resume_contract(previous, current, allow_adapter_updates=True)), 2)
        self.assertEqual(previous['code']['/test/mcp_servers/ufo/stage_runner.py'], 'before')
        current['code']['/model/source/ufo/model.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'external model'):
            check_resume_contract(previous, current, allow_adapter_updates=True)

    def test_resume_rejects_changed_input_and_checkpoint(self):
        for key, value in [('sha256', 'other-video'), ('checkpoint', {'path': '/other.pth'})]:
            previous = self.contract()
            current = copy.deepcopy(previous)
            current[key] = value
            with self.assertRaisesRegex(ValueError, key):
                check_resume_contract(previous, current, allow_adapter_updates=True)

    def test_missing_or_empty_artifact_is_not_a_completed_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            marker, result = Path(folder) / 'done.json', Path(folder) / 'result.pt'
            marker.write_text('{}')
            self.assertFalse(stage_complete(marker, [result]))
            result.touch()
            self.assertFalse(stage_complete(marker, [result]))
            result.write_bytes(b'cached artifact')
            self.assertTrue(stage_complete(marker, [result]))
            self.assertEqual(result.read_bytes(), b'cached artifact')

    def test_completed_preprocessing_is_not_reexecuted_before_ufo_retry(self):
        from mcp_servers.ufo.offline_pipeline import run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            workspace = root / 'scene'
            inputs = workspace / 'input'
            logs = workspace / 'logs/offline'
            def file(path, data=b'cached'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                return path
            resources = {key: root / key for key in ('legacy', 'source', 'worldengine', 'install')}
            for name in ('prepare_scene.py', 'build_scene100.py', 'assemble_scene.py'):
                file(resources['legacy'] / name)
            for name in ('export_ufo_background.py', 'package_ufo_asset.py', 'render_ufo_replay.py'):
                file(resources['worldengine'] / 'scripts' / name)
            video, checkpoint = file(root / 'video.mp4'), file(root / 'model.pth')
            for name in ('decode', 'cameras', 'sam', 'adapter', 'semantics'):
                file(logs / f'{name}.done.json', b'{}')
            for name in ('input_manifest.json', 'camera_manifest.json', 'geometry_config.json', 'annotations.txt',
                         'scene.json', 'sam.done.json', 'cameras/scenes/video/done.json',
                         'cameras/global_aligned/video/omega_pose_override.npz', 'ufo_cameras/video/omega_pose_override.npz'):
                file(inputs / name)
            for i in range(40):
                file(inputs / f'rgb/video/0/{i:06d}.jpg')
                file(inputs / f'sam/video/0/mask_data/mask_{i:06d}.npy')
                file(inputs / f'data/datasets/waymo/video/images/{i:06d}_0.jpg')
                file(workspace / f'reconstruction/000/masks/{i:06d}.npy')
            file(workspace / 'reconstruction/000/PREPARED.json')
            file(workspace / 'OFFLINE_CONTRACT.json', b'{}')
            with patch('mcp_servers.ufo.offline_pipeline.resources', return_value=resources), \
                 patch('mcp_servers.ufo.offline_pipeline.python_path', return_value=Path('/ufoex/python')), \
                 patch('mcp_servers.ufo.offline_pipeline.offline_env', return_value={}), \
                 patch('mcp_servers.ufo.service.REQUESTED_CHECKPOINT', checkpoint), \
                 patch('mcp_servers.ufo.resume.check_resume_contract', return_value=['adapter repair']), \
                 patch('mcp_servers.ufo.offline_pipeline.extract_input', side_effect=AssertionError('decode was rerun')), \
                 patch('mcp_servers.ufo.offline_pipeline.subprocess.run', return_value=SimpleNamespace(returncode=1)) as execute:
                with self.assertRaisesRegex(RuntimeError, 'ufo failed'):
                    run(video, workspace, 0, resume=True)
            self.assertEqual(execute.call_count, 1)
            self.assertTrue(any(str(part).endswith('build_scene100.py') for part in execute.call_args.args[0]))
            self.assertEqual((inputs / 'rgb/video/0/000000.jpg').read_bytes(), b'cached')


if __name__ == '__main__':
    unittest.main()
