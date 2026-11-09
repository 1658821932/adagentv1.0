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
    def test_expert_rear_axle_roundtrip_and_attack_actors_remain_in_world(self):
        np=self.np;frames=self.frames();times,xy,yaw=self.exporter.ego_path_from_frames(frames)
        edited=xy.copy();edited[times>3,0]-=.25
        attack,_=self.exporter.build(frames,{'CAM_F0':{}},'scene','real-asset')
        expert,_=self.exporter.build(frames,{'CAM_F0':{}},'scene','real-asset',expert=(times,edited,yaw))
        self.assertTrue(np.array_equal(attack['object_track']['car']['state']['position'],expert['object_track']['car']['state']['position']))
        center=expert['object_track']['ego']['state']['position'][:,:2];self.assertTrue(np.allclose(center-[1.461,0.],edited))
        source=frames[-1];new=expert['metadata']['openscene_data_infos_dict'][source['token']]
        before=np.asarray(source['ego2global'])@np.r_[source['anns']['gt_boxes'][0][:3],1.]
        after=np.asarray(new['ego2global'])@np.r_[new['anns']['gt_boxes'][0][:3],1.]
        self.assertTrue(np.allclose(before,after));self.assertEqual(frames[-1]['anns']['gt_boxes'][0][0],12.)
        self.assertTrue(np.array_equal(expert['metadata']['digitaltwin_ego2globals'],[f['ego2global'] for f in frames]))
    def test_history_and_nonuniform_times_are_rejected(self):
        np=self.np;frames=self.frames();t,x,y=self.exporter.ego_path_from_frames(frames);x[0,0]+=1
        with self.assertRaisesRegex(ValueError,'immutable'):self.exporter.build(frames,{},'s','a',expert=(t,x,y))
        frames[-1]['timestamp']+=20000
        with self.assertRaisesRegex(ValueError,'Nonuniform'):self.exporter.build(frames,{},'s','a')
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
