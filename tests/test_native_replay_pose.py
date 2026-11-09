"""Replay-only rigid pose override includes static assets and honors explicit poses."""
from types import SimpleNamespace
from mcp_servers.worldengine.adapter.replay_pose import add_static_calibration,explicit_pose_or_log

class Tensor:
    def __init__(self,shape):self.shape=shape
    def reshape(self,*shape):return Tensor(shape)
    def double(self):return self
    def float(self):return self

def test_static_actor_calibration_and_explicit_pose_take_precedence():
    model=SimpleNamespace(exist_log=True,static_in_log=True,log_trans=Tensor((3,)),log_quats=Tensor((4,)))
    renderer=SimpleNamespace(submodel_names={'car':'car','background':'bg'},node_types={'car':'na','background':'background'},gaussian_models={'car':model,'bg':object()})
    result=add_static_calibration(renderer,{'ego':'original'})
    assert result['ego']=='original' and result['car']['translation'].shape==(1,3)
    q,t=Tensor((4,)),Tensor((3,))
    def original(*args,**kwargs):raise AssertionError('Explicit pose must not fall back to source log')
    assert explicit_pose_or_log(original,model,q,t,42)==(q,t,42)

def test_unedited_pose_keeps_original_log_and_dynamic_calibration():
    marker=object();model=SimpleNamespace(exist_log=True,static_in_log=False)
    renderer=SimpleNamespace(submodel_names={'car':'car'},node_types={'car':'na'},gaussian_models={'car':model})
    assert add_static_calibration(renderer,{'car':marker})['car'] is marker
    assert explicit_pose_or_log(lambda *args,**kwargs:'source-log',model,timestamp=42)=='source-log'

def test_install_uses_runtime_manager_and_is_replay_process_local(monkeypatch):
    import sys,types
    from mcp_servers.worldengine.adapter.replay_pose import install
    class Manager:STATIC_THRESHOLD=3
    class Renderer:
        def set_asset(self,asset):pass
        def calibrate_agent_state(self):return {'ego':'source'}
    class Rigid:
        def get_gaussian_rgbs(self,**kwargs):return 'source-colors'
        def _decide_global_pose(self,**kwargs):return 'source'
    modules={
        'worldengine.manager.agent_manager':dict(BaseAgentManager=Manager),
        'worldengine.render.mtgs.mtgs':dict(MTGSRenderEngine=Renderer,rasterization=lambda **kwargs:None),
        'worldengine.render.mtgs.gaussian_model.rigid_object':dict(RigidPortableSubModel=Rigid)}
    for name,attrs in modules.items():
        module=types.ModuleType(name)
        for key,value in attrs.items():setattr(module,key,value)
        monkeypatch.setitem(sys.modules,name,module)
    install();assert Manager.STATIC_THRESHOLD==-1.
    renderer=Renderer();renderer.submodel_names={};assert renderer.calibrate_agent_state()=={'ego':'source'}
    q,t=Tensor((4,)),Tensor((3,));assert Rigid()._decide_global_pose(q,t,42)==(q,t,42)
    first=Renderer.calibrate_agent_state;install();assert Renderer.calibrate_agent_state is first

def test_added_gaussian_token_preserves_original_track_with_underscores():
    from mcp_servers.worldengine.adapter.replay_pose import bind_added_tokens
    model=object();renderer=SimpleNamespace(node_types={'abc':'na'},model_types={'abc':'rigid'},submodel_names={'abc':'model-abc'},gaussian_models={'model-abc':model})
    bind_added_tokens(renderer,[dict(render_token='abc',track_id='synthetic_vehicle_001')])
    assert renderer.submodel_names=={'synthetic_vehicle_001':'model-abc'}
    assert renderer.gaussian_models[renderer.submodel_names['synthetic_vehicle_001']] is model
    assert renderer.model_types['synthetic_vehicle_001']=='rigid'

def test_global_ego_calibration_subtracts_all_three_reconstruction_coordinates():
    import numpy as np
    from mcp_servers.worldengine.adapter.replay_pose import shift_ego_calibration
    class Array(np.ndarray):
        def new_tensor(self,value):return np.asarray(value)
    translation=np.array([[588030.,4475708.,197.47]]).view(Array)
    renderer=SimpleNamespace(engine=SimpleNamespace(current_scene={'metadata':{'native_export':{'ego_calibration_frame':'global'}}}),recon2global_translation=np.array([588000.,4475700.,197.]))
    result=shift_ego_calibration(renderer,{'ego':{'translation':translation}})
    np.testing.assert_allclose(result['ego']['translation'],[[30.,8.,.47]])
    np.testing.assert_array_equal(translation,[[588030.,4475708.,197.47]])
