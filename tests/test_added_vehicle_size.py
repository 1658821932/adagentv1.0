"""Internal replay must use the same scaled Gaussian geometry as native export."""
import ast
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest

CORE=Path(__file__).resolve().parents[1]/"mcp_servers/scene_editing/core/adversarial"

@pytest.fixture
def numeric_renderer(tmp_path):
    np=pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    sys.path.insert(0,str(CORE))
    from native_vehicle_asset import resize_gaussians
    from scipy.spatial.transform import Rotation
    q=Rotation.from_euler("z",.4).as_quat()[[3,0,1,2]]
    data=dict(means=np.array([[-1.,-2.,-3.],[1.,2.,3.]],dtype=np.float32),
              dimensions=np.array([2.,4.,6.]),quats=np.array([q*2,q*2]),
              scales=np.log(np.array([[.2,.4,.6],[.3,.5,.7]])),
              dc=np.zeros((2,3)),rest=np.zeros((2,15,3)),opacity=np.zeros((2,1)))
    reader=Mock(return_value=data)
    class Capture:
        def __init__(self,data,center,rotation):
            self.data,self.center,self.rotation=data,center,rotation
    ns=dict(np=np,Path=Path,lru_cache=lru_cache,read_vehicle_ply=reader,
            resize_gaussians=resize_gaussians,AddedVehicle=Capture)
    tree=ast.parse((CORE/"render_added_vehicle_mtgs.py").read_text())
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)
               and node.name in ("scaled_vehicle_ply","added_vehicle_model")]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(CORE/"render_added_vehicle_mtgs.py"),"exec"),ns)
    asset=tmp_path/"vehicle.ply";asset.write_bytes(b"fixture")
    class Road:
        calls=[]
        def placement_plane(self,x,y,length,width,yaw):
            self.calls.append((x,y,length,width,yaw))
            return dict(z_center=3.,rotation=Rotation.from_euler("z",yaw).as_matrix(),
                        max_corner_residual_m=0.,pitch=0.,roll=0.)
    return ns,asset,data,reader,Road()

def test_replay_matches_native_resize_and_requested_height(numeric_renderer):
    import numpy as np
    from scipy.spatial.transform import Rotation
    ns,asset,data,reader,road=numeric_renderer
    original_q=data["quats"].copy()
    expected=ns["resize_gaussians"](data,[4.,2.,9.])
    model,report=ns["added_vehicle_model"](
        dict(asset_path=str(asset),size_m=[4.,2.,9.]),dict(x=7.,y=-2.),np.eye(4),road,.7)
    np.testing.assert_allclose(model.data["means"].max(0)-model.data["means"].min(0),[4.,2.,9.])
    np.testing.assert_allclose(model.data["means"],expected[0],atol=1e-6)
    np.testing.assert_allclose(model.center,[7.,-2.,7.5])
    expected_R=Rotation.from_euler("z",.7).as_matrix()
    np.testing.assert_allclose(model.rotation,expected_R)
    def covariance(log_scales,quats):
        r=Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
        return np.einsum("nij,nj,nkj->nik",r,np.exp(log_scales)**2,r)
    np.testing.assert_allclose(covariance(model.data["scales"],model.data["quats"]),
                               covariance(expected[1],expected[2]),atol=2e-6)
    np.testing.assert_array_equal(data["quats"],original_q)
    assert model.data["means"].dtype==np.float32
    assert model.data["scales"].dtype==np.float32
    assert model.data["quats"].dtype==np.float32
    assert model.data["opacity"].shape==(2,)
    assert report["requested_dimensions_m"]==[4.,2.,9.]
    assert report["original_asset_dimensions_m"]==[2.,4.,6.]
    assert road.calls[-1][2:]==(4.,2.,.7)

def test_cache_is_bound_to_asset_version_and_size_but_not_pose(numeric_renderer):
    import numpy as np
    ns,asset,data,reader,road=numeric_renderer
    info=dict(asset_path=str(asset),size_m=[4.,2.,9.])
    one,_=ns["added_vehicle_model"](info,dict(x=0.,y=0.),np.eye(4),road,0.)
    two,_=ns["added_vehicle_model"](info,dict(x=2.,y=1.),np.eye(4),road,.5)
    assert reader.call_count==1
    assert one.data is two.data
    assert not np.array_equal(one.center,two.center)
    ns["added_vehicle_model"]({**info,"size_m":[2.,4.,6.]},dict(x=0.,y=0.),np.eye(4),road,0.)
    assert reader.call_count==2
    asset.write_bytes(b"changed asset version")
    ns["added_vehicle_model"](info,dict(x=0.,y=0.),np.eye(4),road,0.)
    assert reader.call_count==3

@pytest.mark.parametrize("size",[[1.,0.,2.],[1.,2.],[float("nan"),2.,3.],[1.,2.,float("inf")]])
def test_invalid_dimensions_are_rejected_before_loading_asset(numeric_renderer,size):
    import numpy as np
    ns,asset,data,reader,road=numeric_renderer
    with pytest.raises(ValueError):
        ns["added_vehicle_model"](dict(asset_path=str(asset),size_m=size),dict(x=0.,y=0.),np.eye(4),road,0.)
    reader.assert_not_called()
