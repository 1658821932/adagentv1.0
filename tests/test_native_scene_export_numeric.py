"""Native export binds edited poses and real source calibrations without changing assets."""
import sys,unittest,importlib.util,json,tempfile
from pathlib import Path
CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
@unittest.skipUnless(importlib.util.find_spec('shapely'),'requires geometry environment')
class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(CORE))
        import numpy as np
        import native_scene_export
        cls.np=np;cls.exporter=native_scene_export
    def frames(self):
        np=self.np;frames=[]
        for i in range(9):
            pose=np.eye(4);pose[:2,3]=[100+i,200]
            frames.append(dict(token='t'+str(i),timestamp=1000000+i*500000,scene_token='source',map_location='map',ego2global=pose.tolist(),ego2global_rotation=[1.,0.,0.,0.],lidar2ego=np.eye(4).tolist(),lidar2ego_rotation=[1,0,0,0],lidar2ego_translation=[0,0,0],cams={'CAM_F0':{}},traffic_lights=[],anns=dict(track_tokens=['car'],gt_names=['vehicle'],gt_boxes=[[20-i,2.,0.,4.,2.,1.5,0.]],gt_velocity_3d=[[0.,0.,0.]])))
        return frames
    def calibration(self,frames):
        from reconstruction_calibration import ReconstructionCalibration
        poses=self.np.asarray([f['ego2global'] for f in frames]).copy()
        poses[:,:2,3]-=[100.,200.];poses[:,2,3]=.35
        return ReconstructionCalibration(poses,[100.,200.,0.],{'source':'offline reconstruction fixture'})
    def test_expert_rear_axle_roundtrip_and_attack_actors_remain_in_world(self):
        np=self.np;frames=self.frames();times,xy,yaw=self.exporter.ego_path_from_frames(frames)
        edited=xy.copy();edited[times>3,0]-=.25
        calibration=self.calibration(frames)
        attack,_=self.exporter.build(frames,{'CAM_F0':{}},'scene','real-asset',reconstruction=calibration)
        expert,_=self.exporter.build(frames,{'CAM_F0':{}},'scene','real-asset',expert=(times,edited,yaw),reconstruction=calibration)
        self.assertTrue(np.array_equal(attack['object_track']['car']['state']['position'],expert['object_track']['car']['state']['position']))
        center=expert['object_track']['ego']['state']['position'][:,:2];self.assertTrue(np.allclose(center-[1.461,0.],edited))
        source=frames[-1];new=expert['metadata']['openscene_data_infos_dict'][source['token']]
        before=np.asarray(source['ego2global'])@np.r_[source['anns']['gt_boxes'][0][:3],1.]
        after=np.asarray(new['ego2global'])@np.r_[new['anns']['gt_boxes'][0][:3],1.]
        self.assertTrue(np.allclose(before,after));self.assertEqual(frames[-1]['anns']['gt_boxes'][0][0],12.)
        self.assertTrue(np.array_equal(expert['metadata']['digitaltwin_ego2globals'],calibration.poses))
        self.assertEqual(expert['metadata']['native_export']['ego_calibration_frame'],'reconstruction')
    def test_initial_pose_and_nonuniform_times_are_rejected(self):
        np=self.np;frames=self.frames();t,x,y=self.exporter.ego_path_from_frames(frames);x[0,0]+=1
        with self.assertRaisesRegex(ValueError,'initial scene pose'):self.exporter.build(frames,{},'s','a',expert=(t,x,y))
        frames[-1]['timestamp']+=20000
        with self.assertRaisesRegex(ValueError,'Nonuniform'):self.exporter.build(frames,{},'s','a')

    def test_millisecond_jitter_preserves_poses_and_frames_on_one_nominal_clock(self):
        np=self.np;frames=self.frames()
        for frame,offset in zip(frames,[0,221,-71,-449,-905,-1200,-1500,-1788,-947]):
            frame['timestamp']+=offset
        original=json.loads(json.dumps(frames))
        calibration=self.calibration(frames)
        scene,checks=self.exporter.build(frames,{'CAM_F0':{}},'s','a',reconstruction=calibration)
        exported=scene['metadata']['openscene_data_infos_dict']
        self.assertEqual(frames,original)
        self.assertEqual([exported[f['token']]['timestamp'] for f in frames],list(range(1000000,5500000,500000)))
        self.assertTrue(np.allclose(scene['object_track']['ego']['state']['velocity'][:,0],2.))
        self.assertTrue(np.array_equal(scene['metadata']['digitaltwin_ego2globals'],calibration.poses))
        self.assertAlmostEqual(checks['max_uniform_grid_drift_s'],.001788)
        self.assertTrue(checks['time_alignment']['poses_and_sensor_frames_preserved'])
        self.assertEqual(scene['metadata']['native_export']['original_timestamps_us'],[f['timestamp'] for f in frames])
        self.assertTrue(np.allclose(scene['metadata']['native_export']['recorded_ego_time_s'],np.arange(9)*.5))

    def test_missing_frame_is_not_fabricated_by_alignment(self):
        frames=self.frames();del frames[4]
        with self.assertRaisesRegex(ValueError,'Nonuniform'):
            self.exporter.build(frames,{},'s','a')

    def test_jittered_takeover_pose_is_locked_in_edit_and_native_export(self):
        from ego_trajectory_edit import modern_edit
        from mcp_servers.scene_editing.shared_history import shared_history_end_index
        np=self.np
        for jitter_us in (528, 2000):
            with self.subTest(jitter_us=jitter_us):
                frames=self.frames()
                frames[3]['timestamp']+=jitter_us
                times,xy,yaw=self.exporter.ego_path_from_frames(frames)
                prefix_end=shared_history_end_index(times)
                self.assertEqual(prefix_end,3)
                operation=dict(edit_mode='speed_schedule',operation='set_speed_targets',
                    parent_candidate_id='recorded_baseline',start_time_s=1.5,reason='Slow down',
                    targets=[dict(time_s=3.5,speed_mps=0.)])
                edited_xy,edited_yaw,curve=modern_edit(times,lambda key:(xy,yaw),operation,float(times[prefix_end]))
                self.assertEqual(curve['actual_start_state']['time_s'],times[3])
                np.testing.assert_allclose(edited_xy[:4],xy[:4],atol=1e-8,rtol=0)
                self.assertFalse(np.allclose(edited_xy[4:],xy[4:]))
                calibration=self.calibration(frames)
                attack,_=self.exporter.build(frames,{},'s','a',reconstruction=calibration)
                expert,checks=self.exporter.build(frames,{},'s','a',expert=(times,edited_xy,edited_yaw),
                    reconstruction=calibration)
                self.assertEqual(checks['time_alignment']['shared_history_end_frame_index'],3)
                for key in ('position','heading','velocity'):
                    np.testing.assert_allclose(expert['object_track']['ego']['state'][key][:4],
                        attack['object_track']['ego']['state'][key][:4],atol=1e-8,rtol=0)
                edited_xy[3,0]+=.00011
                with self.assertRaisesRegex(ValueError,'shared 0–1.5s history'):
                    self.exporter.build(frames,{},'s','a',expert=(times,edited_xy,edited_yaw),
                        reconstruction=calibration)

    def test_expert_edits_after_shared_history_have_consistent_actor_coordinates(self):
        np=self.np;frames=self.frames();times,xy,yaw=self.exporter.ego_path_from_frames(frames)
        edited=xy.copy();edited[times>1.5,0]-=.2*np.minimum(times[times>1.5]-1.5,2.)
        scene,checks=self.exporter.build(frames,{'CAM_F0':{}},'scene','asset',expert=(times,edited,yaw),reconstruction=self.calibration(frames))
        centres=scene['object_track']['ego']['state']['position'][:,:2]
        self.assertTrue(np.allclose(centres-[1.461,0.],edited))
        for original in frames:
            generated=scene['metadata']['openscene_data_infos_dict'][original['token']]
            before=np.asarray(original['ego2global'])@np.r_[original['anns']['gt_boxes'][0][:3],1.]
            after=np.asarray(generated['ego2global'])@np.r_[generated['anns']['gt_boxes'][0][:3],1.]
            self.assertTrue(np.allclose(before,after))
    def test_raw_utm_pose_cannot_be_used_as_an_implicit_render_calibration(self):
        with self.assertRaisesRegex(ValueError,'reconstruction calibration is required'):
            self.exporter.build(self.frames(),{},'scene','asset')
    def test_missing_assets_return_explicit_unavailable_instead_of_original_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'metadata').mkdir();(root/'metadata/frames.json').write_text(json.dumps(self.frames()));(root/'metadata/cameras.json').write_text('{}')
            report=self.exporter.export(dict(sample_root=str(root),scene_id='s'),root,root/'native')
            self.assertEqual(report['status'],'unavailable');self.assertIsNone(report['simulation_input']);self.assertFalse((root/'native/all_scenarios.pkl').exists())
if __name__=='__main__':unittest.main()

@unittest.skipUnless(importlib.util.find_spec('shapely'),'requires geometry environment')
class MapBindingTests(unittest.TestCase):
    def test_source_topology_is_retained_and_unknown_transforms_are_rejected(self):
        sys.path.insert(0,str(CORE));import numpy as np
        from native_scene_map import align
        source={};official=dict(first_frame_yaw_rad=0.,features={})
        for i in range(2):
            polygon=np.array([[i*10.,0.],[i*10.+4.,0.],[i*10.+4.,2.],[i*10.,2.]])
            source[str(i)]=dict(type='LANE_SURFACE_STREET',polygon=polygon.tolist(),polyline=[[i*10.,1.],[i*10.+4.,1.]],roadblock_id='block',entry_lanes=['prev'],exit_lanes=['next'])
            official['features'][str(i)]=dict(token=str(i),polygon=(polygon+[1.,2.]).tolist())
        result,evidence=align(source,official)
        self.assertEqual(result['0']['entry_lanes'],['prev']);self.assertTrue(np.array_equal(result['0']['polyline'],[[1.,3.],[5.,3.]]))
        self.assertEqual(source['0']['polyline'],[[0.,1.],[4.,1.]])
        self.assertLess(evidence['max_polygon_hausdorff_error_m'],1e-12)
        official['features']['1']['polygon']=(np.array(official['features']['1']['polygon'])+[.1,0.]).tolist()
        with self.assertRaisesRegex(ValueError,'coordinate binding'):align(source,official)
