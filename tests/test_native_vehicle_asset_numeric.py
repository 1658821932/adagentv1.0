"""Physical size changes must also change Gaussian ellipsoids, not only point centres."""
import importlib.util,sys,unittest
from pathlib import Path
CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
@unittest.skipUnless(importlib.util.find_spec('scipy'),'requires numeric environment')
class AddedAssetTests(unittest.TestCase):
    def test_resizing_matches_box_and_transforms_rotated_covariance(self):
        sys.path.insert(0,str(CORE));import numpy as np
        from scipy.spatial.transform import Rotation
        from native_vehicle_asset import resize_gaussians
        quat=Rotation.from_euler('z',.4).as_quat()[[3,0,1,2]]
        data=dict(means=np.array([[-1.,-2.,-3.],[1.,2.,3.]]),dimensions=np.array([2.,4.,6.]),quats=np.array([quat,quat]),scales=np.log(np.array([[.2,.4,.6],[.3,.5,.7]])))
        means,scales,quats,stretch=resize_gaussians(data,[4.,2.,9.])
        np.testing.assert_allclose(means.max(0)-means.min(0),[4.,2.,9.])
        old=Rotation.from_quat(data['quats'][:,[1,2,3,0]]).as_matrix();new=Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
        old_cov=np.einsum('nij,nj,nkj->nik',old,np.exp(data['scales'])**2,old)
        new_cov=np.einsum('nij,nj,nkj->nik',new,np.exp(scales)**2,new)
        np.testing.assert_allclose(new_cov,old_cov*stretch[None,:,None]*stretch[None,None,:],atol=1e-12)
        for size in [[1.,0.,1.],[1.,2.],[float('nan'),2.,3.]]:
            with self.assertRaises(ValueError):resize_gaussians(data,size)
if __name__=='__main__':unittest.main()
