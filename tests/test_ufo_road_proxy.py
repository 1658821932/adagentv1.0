from pathlib import Path
import tempfile
import unittest
try:
    import numpy as np
    import torch
except ImportError:
    np=torch=None


@unittest.skipIf(np is None or torch is None,'Road proxy needs numpy/torch')
class RoadProxyTests(unittest.TestCase):
    def test_only_static_road_moves_and_source_projection_is_preserved(self):
        from mcp_servers.ufo.road_proxy import repair_road
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'reconstruction/000/pred100k/sources';folder.mkdir(parents=True)
            classes=np.zeros((10,10),int);classes[7,6]=13
            np.savez(folder/'000000.npz',classes=classes,dynamic=classes==13,coverage=np.zeros((10,10)))
            state={'gauss_params.means':torch.tensor([[0.,2,10],[1,2,10],[0,-1,10]]),
                   'gauss_params.scales':torch.full((3,3),float(np.log(.1))),
                   'gauss_params.opacities':torch.zeros(3,1),'source_time_s':torch.zeros(3)}
            original=state['gauss_params.means'].clone()
            K=np.array([[10.,0,5],[0,10,5],[0,0,1]])
            result=repair_road(state,root,np.stack([np.eye(4)]),np.stack([K]),np.array([0.]),
                               {'normal_world':[0,-1,0],'offset_native':1})
            self.assertTrue(torch.allclose(state['gauss_params.means'][0],torch.tensor([0.,1,5])))
            self.assertTrue(torch.equal(original[1],state['gauss_params.means'][1]))
            self.assertTrue(torch.allclose(state['gauss_params.scales'][0].exp(),torch.full((3,),.05)))
            self.assertEqual(float(state['gauss_params.opacities'][2].sigmoid()),0)
            self.assertEqual(result['road_gaussians_repositioned'],1)
            self.assertEqual(result['invalid_road_intersections_removed'],1)
            self.assertLess(result['max_source_projection_error_pixels'],1e-5)


if __name__=='__main__':unittest.main()
