import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Unit-contract tests require numpy')
class FrontendUnitTests(unittest.TestCase):
    def fixture(self, root):
        workspace = root/'scene'
        frontend = workspace/'input/cameras/global_aligned/video'
        frontend.mkdir(parents=True)
        metadata = workspace/'input/cameras/scenes/video/metadata.json'
        metadata.parent.mkdir(parents=True)
        metadata.write_text(json.dumps({'camera_basis':'OpenCV: right,down,forward',
                                        'metric_scale_source':'MoGe-2/GCA estimated metric scale'}))
        upstream = np.tile(np.eye(4), (40,1,1))
        upstream[:,0,3] = np.linspace(0,73,40)
        K = np.tile([[400.,0,260],[0,400,176],[0,0,1]], (40,1,1))
        poses = frontend/'omega_pose_override.npz'
        np.savez(poses, coordinate_frame=np.asarray('global_metric'),
                 metric_scale_source=np.asarray('moge2_gca_plus_overlap_camera_se3'),
                 frame_ids=np.arange(40),camera_ids=np.array(['0']*40),
                 omega_camera_to_world_global_metric=upstream,predicted_intrinsics_ufo=K)
        transform = np.array([[0,0,1,3],[-1,0,0,4],[0,-1,0,5],[0,0,0,1.]])
        target = root/'assets_root/assets/scene/cameras.npz'
        target.parent.mkdir(parents=True)
        np.savez(target,C=transform[None]@upstream,K=K)
        return workspace,root/'assets_root',target

    def test_rigid_export_preserves_estimated_meter_without_claiming_depth_accuracy(self):
        from mcp_servers.ufo.frontend_units import verify_frontend_units
        with tempfile.TemporaryDirectory() as folder:
            workspace,assets,target=self.fixture(Path(folder))
            report=verify_frontend_units(workspace,assets)
            self.assertEqual(report['metres_per_native_unit'],1.)
            self.assertFalse(report['absolute_accuracy_certified'])
            self.assertEqual(report['validation_scope'],'upstream_estimated_coordinate_units')
            self.assertLess(report['max_camera_translation_error_m'],1e-10)

    def test_scaled_export_or_changed_camera_identity_is_rejected(self):
        from mcp_servers.ufo.frontend_units import verify_frontend_units
        with tempfile.TemporaryDirectory() as folder:
            workspace,assets,target=self.fixture(Path(folder))
            with np.load(target) as source:
                C,K=source['C'].copy(),source['K'].copy()
            C[:,:3,3]*=2
            np.savez(target,C=C,K=K)
            with self.assertRaisesRegex(ValueError,'same metric camera sequence'):
                verify_frontend_units(workspace,assets)

    def test_motion_ground_plane_recovers_pitch_without_claiming_global_surface(self):
        from mcp_servers.ufo.metric_scene import motion_ground_plane
        theta=.15
        tangent=np.array([np.cos(theta),0,-np.sin(theta)])
        normal=np.array([np.sin(theta),0,np.cos(theta)])
        poses=np.tile(np.eye(4),(40,1,1));poses[:,:3,3]=np.linspace(0,70,40)[:,None]*tangent
        points=np.array([-1.5*normal+x*tangent+np.array([0,y,0]) for x in np.linspace(6,8,20) for y in np.linspace(-2,2,20)])
        result=motion_ground_plane(poses,{'normal_world':[0,0,1]},points,1.)
        self.assertTrue(np.allclose(result['normal_world'],normal))
        self.assertAlmostEqual(result['offset_native'],1.5)
        self.assertFalse(result['global_road_surface_validated'])
        self.assertEqual(result['inlier_fraction'],1.)
        poses[:,1,3]=np.sin(np.linspace(0,np.pi,40))*20
        with self.assertRaisesRegex(ValueError,'straight driving'):
            motion_ground_plane(poses,{'normal_world':[0,0,1]},points,1.)
