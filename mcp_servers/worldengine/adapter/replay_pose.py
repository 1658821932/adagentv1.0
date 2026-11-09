"""Exact trajectory replay must also move rigid assets reconstructed as static.

Installed only inside the standalone native-replay subprocess. Source WorldEngine
classes/checkpoints are never rewritten, and policy evaluation is unaffected.
"""

def add_static_calibration(renderer, calibration):
    for token, name in renderer.submodel_names.items():
        model=renderer.gaussian_models[name]
        if renderer.node_types[token] != 'na' or not getattr(model,'exist_log',False) or not model.static_in_log:
            continue
        calibration[token]={'translation':model.log_trans.reshape(1,3).double(),
                            'rotation':model.log_quats.reshape(1,4).double()}
    return calibration


def shift_ego_calibration(renderer,calibration):
    """Our exported original ego matrices are global; MTGS pose queries are local."""
    scene=getattr(getattr(renderer,'engine',None),'current_scene',{})
    exported=scene.get('metadata',{}).get('native_export')
    if exported and exported.get('ego_calibration_frame','global')=='global':
        translation=calibration['ego']['translation']
        calibration['ego']['translation']=translation-translation.new_tensor(renderer.recon2global_translation)
    return calibration


def bind_added_tokens(renderer, bindings):
    """WE checkpoint suffix tokens are opaque; retain the public scene track IDs."""
    for item in bindings:
        token=item['render_token'];track=item['track_id']
        if token not in renderer.submodel_names:raise ValueError('Added scene vehicle has no Gaussian model: '+track)
        if track!=token and track in renderer.submodel_names:raise ValueError('Added Gaussian track mapping collides: '+track)
        for table in (renderer.node_types,renderer.model_types,renderer.submodel_names):
            table[track]=table.pop(token)


def explicit_pose_or_log(original, model, quat=None, trans=None, timestamp=None):
    if quat is not None and trans is not None:
        if tuple(quat.shape)!=(4,) or tuple(trans.shape)!=(3,):
            raise ValueError('Native replay rigid pose must contain quaternion[4] and translation[3]')
        return quat.float(),trans.float(),timestamp
    return original(model,quat=quat,trans=trans,timestamp=timestamp)


def rasterize_cameras_sequentially(rasterization, **kwargs):
    """Avoid a multi-camera intersection buffer larger than GPU memory."""
    views=kwargs['viewmats'];count=len(views)
    if count<=1:return rasterization(**kwargs)
    import torch
    images=[];alphas=[]
    for i in range(count):
        single=dict(kwargs,viewmats=views[i:i+1],Ks=kwargs['Ks'][i:i+1])
        colors=kwargs['colors']
        if colors.ndim==3 and colors.shape[0]==count:single['colors']=colors[i:i+1]
        if kwargs.get('backgrounds') is not None:single['backgrounds']=kwargs['backgrounds'][i:i+1]
        image,alpha,_=rasterization(**single);images.append(image);alphas.append(alpha)
    return torch.cat(images,dim=0),torch.cat(alphas,dim=0),{}


def install(renderer_only=False):
    if not renderer_only:
        from worldengine.manager.agent_manager import BaseAgentManager
    from worldengine.render.mtgs.mtgs import MTGSRenderEngine
    import importlib
    mtgs=importlib.import_module('worldengine.render.mtgs.mtgs')
    from worldengine.render.mtgs.gaussian_model.rigid_object import RigidPortableSubModel
    if getattr(MTGSRenderEngine,'_native_replay_pose_installed',False):return
    original_raster=mtgs.rasterization
    mtgs.rasterization=lambda **kwargs:rasterize_cameras_sequentially(original_raster,**kwargs)
    original_asset=MTGSRenderEngine.set_asset
    def set_asset(renderer,asset):
        original_asset(renderer,asset)
        scene=renderer.engine.current_scene
        bind_added_tokens(renderer,scene.get('metadata',{}).get('native_export',{}).get('added_vehicle_assets',[]))
    MTGSRenderEngine.set_asset=set_asset
    original=MTGSRenderEngine.calibrate_agent_state
    def calibrate(renderer):
        calibration=shift_ego_calibration(renderer,original(renderer))
        return add_static_calibration(renderer,calibration)
    decide=RigidPortableSubModel._decide_global_pose
    def pose(model,quat=None,trans=None,timestamp=None):
        result=explicit_pose_or_log(decide,model,quat,trans,timestamp)
        if getattr(getattr(model,'config',None),'native_added',False):model._native_replay_quaternion=result[0]
        return result
    original_rgb=RigidPortableSubModel.get_gaussian_rgbs
    def rgb(model,camera_to_worlds,timestamp,device=None):
        if not getattr(getattr(model,'config',None),'native_added',False):return original_rgb(model,camera_to_worlds=camera_to_worlds,timestamp=timestamp,device=device)
        import torch
        from gsplat.cuda._wrapper import spherical_harmonics
        from worldengine.render.mtgs.utils.gaussian_utils import quat_to_rotmat
        rotation=quat_to_rotmat(model._native_replay_quaternion[None])[0]
        view=(model.global_means-camera_to_worlds[..., :3,3])@rotation
        view=torch.nn.functional.normalize(view,dim=-1)
        colors=torch.cat((model.features_dc[:,None,:],model.features_rest),dim=1)
        return torch.clamp(spherical_harmonics(model.sh_degree,view,colors)+.5,0.,1.)
    RigidPortableSubModel.get_gaussian_rgbs=rgb
    MTGSRenderEngine.calibrate_agent_state=calibrate
    RigidPortableSubModel._decide_global_pose=pose
    # Trajectory participants must update their policy each step and be included
    # in RenderManager's dynamic states even if their displacement is tiny/zero.
    if not renderer_only:BaseAgentManager.STATIC_THRESHOLD=-1.
    MTGSRenderEngine._native_replay_pose_installed=True
