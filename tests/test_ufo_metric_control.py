import copy
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mcp_servers.ufo.metric_config import validate, generate

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


def job(calibration, asset=None):
    operations = []
    if asset:
        operations.append({'operation': 'insert', 'object_id': 'new_car', 'asset_manifest_path': str(asset),
                           'position_m': [20, 3, 0], 'yaw_degrees': 0})
    operations.append({'operation': 'translate', 'target_id': 'new_car' if asset else 'selected',
                       'offset_m': [0, -3, 0], 'start_seconds': .1, 'end_seconds': 1.7})
    return {'backend': 'ufo', 'operation': 'metric_edit', 'units': 'meters',
            'calibration_path': str(calibration), 'operations': operations,
            **({} if asset else {'track_id': 10001})}


class MetricContractTests(unittest.TestCase):
    def test_explicit_metric_units_targets_and_grounding(self):
        config = job('/calibration.json', '/asset.json')
        self.assertEqual(validate(config)['operations'][1]['offset_m'], [0., -3., 0.])
        cases = []
        for key, value in [('units', 'native'), ('reference_frame', 'camera_per_frame')]:
            cases.append({**config, key: value})
        for key, value in [('offset_m', [0, float('nan'), 0]), ('offset_m', [0, 0, 1]),
                           ('target_id', 'unknown'), ('end_seconds', -.1), ('trajectory_mode', 'absolute_pose')]:
            changed = copy.deepcopy(config)
            changed['operations'][1][key] = value
            cases.append(changed)
        changed = copy.deepcopy(config)
        changed['operations'][0]['position_m'][2] = 1
        cases.append(changed)
        for changed in cases:
            with self.subTest(config=changed), self.assertRaises(ValueError):
                validate(changed)
        with self.assertRaisesRegex(ValueError, 'target_selection_path'):
            validate(job('/calibration.json'), agent=True)

    def test_insert_only_generator_needs_no_selected_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            calibration, asset = root/'metric.json', root/'asset.json'
            calibration.write_text('{}')
            asset.write_text('{}')
            config = job(calibration, asset)
            result = generate({**config, 'output_path': str(root/'edit.json')})
            self.assertEqual(result['operations'][0]['object_id'], 'new_car')
            self.assertNotIn('track_id', result)
            with self.assertRaisesRegex(ValueError, 'distinct'):
                generate({**config, 'output_path': str(calibration)})

    def test_agent_routes_calibration_and_metric_insert_without_semantic_selection(self):
        from agent.executor import ToolExecutor
        executor = ToolExecutor()
        executor.reconstruction = {'backend': 'ufo_worldengine_time_v2', 'workspace': '/scene', 'frame_count': 40}
        args = executor.prepare_arguments('scene_editor', {'operation': 'calibrate_metric'})
        self.assertEqual(args['ufo_reconstruction'], executor.reconstruction)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory)/'edit.json'
            config.write_text(json.dumps(job('/calibration.json', '/asset.json')))
            args = executor.prepare_arguments('scene_editor', {'operation': 'metric_edit', 'edit_config_path': str(config)})
            self.assertEqual(args['operation'], 'metric_edit')
            config.write_text(json.dumps(job('/calibration.json')))
            with self.assertRaisesRegex(ValueError, 'target_selection_path'):
                executor.prepare_arguments('scene_editor', {'operation': 'metric_edit', 'edit_config_path': str(config)})


@unittest.skipIf(np is None or torch is None, 'CPU numeric fixtures need numpy and torch')
class MetricGeometryTests(unittest.TestCase):
    def fixture(self, root):
        workspace = root/'scene'
        sources = workspace/'reconstruction/000/pred100k/sources'
        sources.mkdir(parents=True)
        (sources.parent/'BUILD.json').write_text(json.dumps({'all_sources': 40}))
        asset = root/'original/assets/scene'
        (asset/'dynamic').mkdir(parents=True)
        (asset/'background').mkdir()
        rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
        yy, xx = np.mgrid[:10, :20]
        tracks = np.where(xx < 10, np.where(yy < 5, 10001, 10002), 0).astype(np.int32)
        dynamic = tracks > 0
        raw = np.stack([(xx-9.5)*.2, yy*.15, 5+yy], axis=-1).astype(np.float32)
        raw[~dynamic, 1] = 1.5
        poses, files = [], []
        for index in range(40):
            C = np.eye(4, dtype=np.float32)
            C[2, 3] = index*.05
            means = raw.copy()+C[:3, 3]
            np.savez(sources/f'{index:06d}.npz', means=means, scales=np.ones_like(means)*.02,
                     C=C, opacities=np.ones(tracks.shape)*.9, dynamic=dynamic,
                     coverage=np.zeros(tracks.shape), tracks=tracks, classes=np.where(dynamic, 13, 0))
            points = means[dynamic] @ rotation.T
            actor = {'means': torch.from_numpy(points), 'scales': torch.ones(len(points), 3)*.02,
                     'quats': torch.tensor([[1., 0, 0, 0]]*len(points)),
                     'opacities': torch.ones(len(points))*.9, 'colors': torch.ones(len(points), 3)*.5}
            file = asset/'dynamic'/f'{index:06d}.pt'
            torch.save(actor, file)
            files.append(str(file))
            pose = C.copy()
            pose[:3, :3] = rotation
            pose[:3, 3] = rotation @ C[:3, 3]
            poses.append(pose)
        ckpt = asset/'background/scene.ckpt'
        torch.save({'observed_dynamic': {'config': {'actor_files': files}, 'state_dict': {}},
                    'background': {'config': {}, 'state_dict': {'unchanged': torch.ones(3)}}}, ckpt)
        np.savez(asset/'cameras.npz', C=np.stack(poses), timestamps=np.arange(40)*.05,
                 K=np.stack([np.eye(3)]*40), ego=np.stack(poses))
        manifest = {'asset_id': 'scene', 'version': 'ufo-worldengine-time-v2', 'asset_ckpt': str(ckpt),
                    'source_to_world_rotation': rotation.tolist(), 'source_origin': [0, 0, 0]}
        (asset/'ASSET.json').write_text(json.dumps(manifest))
        reference = root/'measured.json'
        reference.write_text(json.dumps({'source': 'measured_reference', 'description': 'synthetic known length',
                                         'native_distance': 2, 'distance_m': 5, 'relative_uncertainty': .01}))
        from mcp_servers.ufo.metric_scene import calibrate
        calibration = root/'metric_scene.json'
        result = calibrate(workspace, root/'original', calibration, reference=reference)
        self.assertAlmostEqual(result['meters_per_native_unit'], 2.5)
        points = np.array([[x, y, z] for x in (-2, 2) for y in (-1, 1) for z in (0, 1.5)], dtype=np.float32)
        np.savez(root/'car.npz', means=points, scales=np.ones((8, 3), dtype=np.float32)*.01,
                 quats=np.tile([1, 0, 0, 0], (8, 1)).astype(np.float32),
                 opacities=np.full(8, .95, dtype=np.float32), colors=np.ones((8, 3), dtype=np.float32)*.5)
        asset_manifest = root/'car.json'
        asset_manifest.write_text(json.dumps({'asset_path': 'car.npz', 'format': 'gaussian_npz',
            'meters_per_unit': 1, 'dimensions_m': [4, 2, 1.5], 'asset_to_flu': np.eye(3).tolist(),
            'ground_contact_point': [0, 0, 0], 'quaternion_order': 'wxyz', 'color_space': 'linear_rgb'}))
        reconstruction = {'backend': 'ufo_worldengine_time_v2', 'frame_count': 40, 'workspace': str(workspace),
                          'asset_root': str(root/'original'), 'rendered_video': '/not-rendered.mp4'}
        return reconstruction, calibration, asset_manifest, tracks

    def test_insert_then_right_three_meters_keeps_size_and_original_actors(self):
        from mcp_servers.ufo.metric_control import execute
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reconstruction, calibration, asset, tracks = self.fixture(root)
            config = job(calibration, asset)
            result = execute(reconstruction, config, root/'edited', render=False)
            self.assertEqual(result['units'], 'meters')
            self.assertEqual(result['profile'], 'rigid_metric')
            destination = Path(result['asset_manifest']).parent
            before = torch.load(root/'original/assets/scene/dynamic/000039.pt', weights_only=True)
            after = torch.load(destination/'dynamic/000039.pt', weights_only=True)
            for key in before:
                self.assertTrue(torch.equal(before[key], after[key][:-8]), key)
            basis = np.asarray(result['road_basis_flu_world'])
            origin = np.asarray(result['road_origin_native'])
            inserted_m = (after['means'][-8:].numpy()-origin) @ basis*2.5
            self.assertTrue(np.allclose(np.ptp(inserted_m, axis=0), [4, 2, 1.5], atol=1e-5))
            self.assertAlmostEqual(float(inserted_m[:, 1].mean()), 0, places=5)
            self.assertAlmostEqual(float(inserted_m[:, 2].min()), 0, places=5)
            self.assertTrue(np.allclose(after['scales'][-8:].numpy(), .004))
            self.assertTrue(np.allclose(result['frames'][-1]['operations'][1]['actual_offset_m'], [0, -3, 0], atol=1e-5))
            content = (destination/'dynamic/000039.pt').read_bytes()
            repeated = execute(reconstruction, config, root/'edited', render=False)
            self.assertEqual(result['job_sha256'], repeated['job_sha256'])
            self.assertEqual(content, (destination/'dynamic/000039.pt').read_bytes())

    def test_existing_track_translation_is_rigid_and_non_targets_unchanged(self):
        from mcp_servers.ufo.metric_control import execute
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reconstruction, calibration, asset, tracks = self.fixture(root)
            result = execute(reconstruction, job(calibration), root/'edited', render=False)
            source = root/'original/assets/scene/dynamic/000039.pt'
            saved = source.read_bytes()
            before = torch.load(source, weights_only=True)
            after = torch.load(Path(result['asset_manifest']).parent/'dynamic/000039.pt', weights_only=True)
            selected = tracks[tracks > 0] == 10001
            for key in before:
                self.assertTrue(torch.equal(before[key][~selected], after[key][~selected]))
                if key != 'means':
                    self.assertTrue(torch.equal(before[key], after[key]))
            delta = (after['means'][selected]-before['means'][selected]).numpy()*2.5
            self.assertTrue(np.allclose(delta, [0, -3, 0], atol=1e-5))
            self.assertEqual(saved, source.read_bytes())

    def test_metric_scale_cache_rejects_changed_reconstruction(self):
        from mcp_servers.ufo.metric_scene import calibrate, load_calibration
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reconstruction, calibration, asset, tracks = self.fixture(root)
            workspace = Path(reconstruction['workspace'])
            cached = calibrate(workspace, root/'original', calibration, reference=root/'measured.json')
            self.assertEqual(cached['scale_fit']['source'], 'measured_reference')
            (workspace/'reconstruction/000/pred100k/BUILD.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'stale'):
                load_calibration(calibration, workspace, root/'original')

    def test_road_frame_removes_camera_pitch_and_translation_keeps_slope_height(self):
        from mcp_servers.ufo.metric_scene import road_frame
        up = np.array([-.1, 0, 1.])
        up /= np.linalg.norm(up)
        pose = np.eye(4)
        pose[:3, 2] = [1, 0, -.3]
        untouched = pose.copy()
        basis, origin = road_frame(pose, {'normal_world': up.tolist(), 'offset_native': 1})
        self.assertTrue(np.array_equal(pose, untouched))
        delta = basis @ np.array([2, -3, 0])
        self.assertAlmostEqual(float(delta @ up), 0)
        self.assertAlmostEqual(float(origin @ up), -1)
        self.assertTrue(np.allclose(basis.T @ basis, np.eye(3)))
        self.assertAlmostEqual(np.linalg.det(basis), 1)

    def test_asset_dimensions_rotation_and_unknown_scale(self):
        from mcp_servers.ufo.metric_assets import load_asset, place_asset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reconstruction, calibration, asset, tracks = self.fixture(root)
            local, metadata = load_asset(asset)
            placed = place_asset(local, np.eye(3), np.zeros(3), [0, 0, 0], 90, 2.5)
            self.assertTrue(np.allclose(np.ptp(placed['means'], axis=0)*2.5, [2, 4, 1.5]))
            self.assertTrue(np.allclose(placed['quats'][0], [2**-.5, 0, 0, 2**-.5]))
            data = json.loads(asset.read_text())
            data.pop('meters_per_unit')
            asset.write_text(json.dumps(data))
            self.assertTrue(np.allclose(load_asset(asset)[0]['means'], local['means']))
            data.pop('dimensions_m')
            asset.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'scale cannot be guessed'):
                load_asset(asset)

    def test_degenerate_or_inconsistent_road_is_rejected(self):
        from mcp_servers.ufo.metric_scene import fit_road_plane
        points = np.zeros((200, 3))
        points[:, 0] = np.arange(200)
        with self.assertRaises(ValueError):
            fit_road_plane(points, 1)
        points = np.random.default_rng(1).uniform(-10, 10, (1000, 3))
        with self.assertRaises(ValueError):
            fit_road_plane(points, 1)

    def test_anisotropic_intrinsics_rectify_moge_input_and_preserve_fov(self):
        from mcp_servers.ufo.metric_edit import moge_input_geometry
        rgb = np.zeros((352, 520, 3), dtype=np.uint8)
        K = np.array([[384.69, 0, 260], [0, 465.48, 176], [0, 0, 1]])
        image, fov = moge_input_geometry(rgb, K)
        self.assertEqual(image.shape, (291, 520, 3))
        self.assertAlmostEqual(fov, float(np.rad2deg(2*np.arctan(260/384.69))))
        bad = K.copy()
        bad[0, 2] += 5
        with self.assertRaisesRegex(ValueError, 'Off-center'):
            moge_input_geometry(rgb, bad)

    def test_ply_activation_and_metric_dimensions(self):
        from mcp_servers.ufo.metric_assets import load_asset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reconstruction, calibration, asset, tracks = self.fixture(root)
            data = np.load(root/'car.npz')
            names = ['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3',
                     'opacity','f_dc_0','f_dc_1','f_dc_2']
            header = 'ply\nformat ascii 1.0\nelement vertex 8\n' + ''.join(f'property float {name}\n' for name in names) + 'end_header\n'
            rows = [' '.join(map(str, [*point, *([float(np.log(.01))]*3), 1,0,0,0, 0,0,0,0])) for point in data['means']]
            (root/'car.ply').write_text(header+'\n'.join(rows)+'\n')
            manifest = json.loads(asset.read_text())
            manifest.update(asset_path='car.ply', format='3dgs_ply', appearance='dc_only')
            asset.write_text(json.dumps(manifest))
            local, metadata = load_asset(asset)
            self.assertTrue(np.allclose(local['scales'], .01))
            self.assertTrue(np.allclose(local['opacities'], .5))
            self.assertTrue(np.allclose(local['colors'], .5))
            self.assertTrue(np.allclose(metadata['dimensions_m'], [4,2,1.5]))

    def test_endpoint_outside_video_does_not_publish_asset(self):
        from mcp_servers.ufo.metric_control import execute
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reconstruction, calibration, asset, tracks = self.fixture(root)
            config = job(calibration)
            config['operations'][0]['end_seconds'] = 3
            with self.assertRaisesRegex(ValueError, 'time span'):
                execute(reconstruction, config, root/'edited', render=False)
            self.assertFalse(list((root/'edited').glob('assets/*')))

    def test_missing_target_at_endpoint_cleans_staging_without_publishing(self):
        from mcp_servers.ufo.metric_control import execute
        from mcp_servers.ufo.metric_scene import calibrate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reconstruction, calibration, asset, tracks = self.fixture(root)
            workspace = Path(reconstruction['workspace'])
            source = workspace/'reconstruction/000/pred100k/sources/000039.npz'
            with np.load(source) as loaded:
                data = {key: loaded[key].copy() for key in loaded.files}
            data['tracks'][data['tracks'] == 10001] = 10002
            np.savez(source, **data)
            calibrate(workspace, root/'original', calibration, reference=root/'measured.json', force=True)
            config = job(calibration)
            config['operations'][0]['end_seconds'] = 1.95
            with self.assertRaisesRegex(ValueError, 'endpoint'):
                execute(reconstruction, config, root/'edited', render=False)
            self.assertFalse(list((root/'edited').glob('assets/*')))
            self.assertFalse(list((root/'edited').glob('.building_*')))

    def test_render_is_reused_after_comparison_failure(self):
        from mcp_servers.ufo.metric_control import execute
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reconstruction, calibration, asset, tracks = self.fixture(root)
            config = job(calibration, asset)
            def render(asset_root, asset_id, output, **kwargs):
                output.write_bytes(b'completed-synthetic-render')
                return {'frames': 40}
            with patch('mcp_servers.ufo.metric_edit.render_asset', side_effect=render) as renderer, \
                 patch('mcp_servers.ufo.video_compare.compare_videos', side_effect=RuntimeError('compare failed')):
                with self.assertRaisesRegex(RuntimeError, 'compare failed'):
                    execute(reconstruction, config, root/'edited')
                self.assertEqual(renderer.call_count, 1)
            with patch('mcp_servers.ufo.metric_edit.render_asset') as renderer, \
                 patch('mcp_servers.ufo.video_compare.compare_videos', return_value={}):
                result = execute(reconstruction, config, root/'edited')
                renderer.assert_not_called()
                self.assertTrue(result['render_complete'])

    def test_actual_offline_service_worker_exports_assets_without_gpu(self):
        from mcp_servers.ufo.service import edit_metric_control
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reconstruction, calibration, asset, tracks = self.fixture(root)
            config = root/'job.json'
            config.write_text(json.dumps(job(calibration, asset)))
            with patch.dict(os.environ, {'UFO_ENV_ROOT': sys.prefix}):
                result = edit_metric_control(reconstruction, edit_config_path=str(config),
                                             output_root=str(root/'worker'), assets_only=True)
            self.assertEqual(result['frame_count'], 40)
            self.assertFalse(result['render_complete'])
            self.assertLess(result['max_execution_error_m'], .001)
            self.assertTrue(Path(result['audit_path']).is_file())
            self.assertNotIn('frames', result)

    def test_frontend_unit_calibration_is_not_a_depth_agreement_certificate(self):
        from mcp_servers.ufo.metric_scene import calibrate, load_calibration, reconstruction_identity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            reconstruction, old_calibration, asset, tracks = self.fixture(root)
            workspace = Path(reconstruction['workspace'])
            metadata = workspace/'input/cameras/scenes/video/metadata.json'
            metadata.parent.mkdir(parents=True)
            metadata.write_text(json.dumps({'camera_basis':'OpenCV: right,down,forward',
                                            'metric_scale_source':'MoGe-2/GCA estimated metric scale'}))
            poses = workspace/'input/cameras/global_aligned/video/omega_pose_override.npz'
            poses.parent.mkdir(parents=True)
            with np.load(root/'original/assets/scene/cameras.npz') as cameras:
                np.savez(poses,coordinate_frame='global_metric',metric_scale_source='moge2_gca_plus_overlap_camera_se3',
                         frame_ids=np.arange(40),camera_ids=np.array(['0']*40),
                         omega_camera_to_world_global_metric=cameras['C'],predicted_intrinsics_ufo=cameras['K'])
            plane = root/'road.json'
            plane.write_text(json.dumps({'normal_world':[0,0,1], 'offset_native':1.5}))
            path = root/'frontend_metric.json'
            identity = reconstruction_identity(workspace,root/'original')
            diagnostic = root/'moge_metric_depths'/identity['fingerprint'][:16]/'rendered_depth_diagnostics/frame_000000.json'
            diagnostic.parent.mkdir(parents=True)
            diagnostic.write_text('{"status":"rejected"}')
            result = calibrate(workspace,root/'original',path,method='frontend_metric',road_plane=plane)
            self.assertEqual(result['meters_per_native_unit'],1)
            self.assertEqual(result['validation_scope'],'upstream_estimated_coordinate_units')
            self.assertFalse(result['scale_fit']['absolute_accuracy_certified'])
            self.assertEqual(result['scale_fit']['independent_depth_reports'],[str(diagnostic)])
            self.assertEqual(result,calibrate(workspace,root/'original',path,method='frontend_metric',road_plane=plane))
            metadata.write_text(metadata.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'provenance changed'):
                load_calibration(path,workspace,root/'original')

    def test_insert_can_follow_ego_translation_without_changing_metric_lateral_offset(self):
        from mcp_servers.ufo.metric_control import execute
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            reconstruction,calibration,asset,tracks=self.fixture(root)
            config=job(calibration,asset)
            config['operations'][0]['motion_reference']='ego_translation'
            result=execute(reconstruction,config,root/'edited',render=False)
            final=result['frames'][-1]['operations']
            self.assertEqual(final[0]['motion_reference'],'ego_translation')
            self.assertGreater(final[0]['reference_offset_flu_m'][0],0)
            self.assertTrue(np.allclose(final[1]['actual_offset_m'],[0,-3,0],atol=1e-4))


if __name__ == '__main__':
    unittest.main()
