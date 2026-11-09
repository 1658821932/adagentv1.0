"""Merge an added Gaussian PLY into a NEW native checkpoint, preserving source data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read_vehicle_ply(path):
    from plyfile import PlyData
    vertices=PlyData.read(str(path))['vertex']
    fields=['x','y','z']+[f'f_dc_{i}' for i in range(3)]+[f'f_rest_{i}' for i in range(45)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    if len(vertices)==0 or not set(fields)<=set(vertices.data.dtype.names):raise ValueError('Added asset requires nonempty degree-3 Gaussian PLY attributes')
    values=np.column_stack([vertices[name] for name in fields]).astype(np.float32)
    if not np.isfinite(values).all():raise ValueError('Added PLY has nonfinite Gaussian values')
    means=values[:,:3];center=(means.min(0)+means.max(0))/2;dimensions=means.max(0)-means.min(0)
    if np.any(dimensions<=0) or np.any(np.linalg.norm(values[:,55:59],axis=1)<1e-8):raise ValueError('Added PLY has degenerate dimensions or quaternion')
    return dict(means=means-center,dimensions=dimensions,dc=values[:,3:6],rest=values[:,6:51].reshape(-1,3,15).transpose(0,2,1).copy(),opacity=values[:,51:52],scales=values[:,52:55],quats=values[:,55:59])


def resize_gaussians(data,size):
    from scipy.spatial.transform import Rotation
    size=np.asarray(size,float)
    if size.shape!=(3,) or np.any(size<=0) or not np.isfinite(size).all():raise ValueError('Added size must be positive finite length/width/height')
    scale=size/data['dimensions'];means=data['means']*scale
    local_quat=np.asarray(data['quats'],float);local_quat/=np.linalg.norm(local_quat,axis=1)[:,None]
    rotations=Rotation.from_quat(local_quat[:,[1,2,3,0]]).as_matrix()
    radii=np.exp(np.asarray(data['scales'],float))
    if not np.isfinite(radii).all():raise ValueError('Added Gaussian scales overflow')
    covariance=np.einsum('nij,nj,nkj->nik',rotations,radii*radii,rotations)
    covariance=covariance*scale[None,:,None]*scale[None,None,:]
    eigen,basis=np.linalg.eigh(covariance)
    if np.any(eigen<=0):raise ValueError('Added Gaussian covariance must stay positive')
    reflected=np.linalg.det(basis)<0;basis[reflected,:,0]*=-1
    q=Rotation.from_matrix(basis).as_quat()[:,[3,0,1,2]]
    return means,.5*np.log(eigen),q,scale


def merge(request):
    import torch
    from render_ego_mtgs import validate_checkpoint_pickle
    from road_height import RoadHeightMap
    source=Path(request['source_checkpoint']);target=Path(request['output_checkpoint'])
    if target.exists():raise FileExistsError('Never overwrite an existing native checkpoint')
    if source.resolve()==target.resolve():raise ValueError('Added asset must not modify the source checkpoint')
    validate_checkpoint_pickle(source)
    asset=torch.load(str(source),map_location='cpu')
    data=read_vehicle_ply(request['asset_path']);size=np.asarray(request['size_m'],float)
    means,log_scales,q,scale=resize_gaussians(data,size)
    from render_added_vehicle_mtgs import rotation_to_quat
    from run_ego_reference import observed_objects_from_frames
    frames=json.loads(Path(request['frames_path']).read_text());track=request['track_id']
    objects=observed_objects_from_frames(frames);first=np.asarray(frames[0]['ego2global']);angle=float(np.arctan2(first[1,0],first[0,0]));rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    recon=np.asarray(asset['background']['config']['recon2world_translation'],float)
    road=RoadHeightMap.load(request['source_scene']);translations=[];quaternions=[];placements=[]
    for frame,actors in zip(frames,objects):
        matches=[a for a in actors if a['track']==track]
        if len(matches)!=1:raise ValueError('Added vehicle must have one frozen pose per native frame')
        actor=matches[0];xy=np.array([actor['x'],actor['y']])@rot.T+first[:2,3]-recon[:2]
        yaw=actor['yaw']+angle;plane=road.placement_plane(*xy,*size[:2],yaw)
        if plane['max_corner_residual_m']>.1:raise ValueError('Added footprint exceeds unchanged road flatness limit')
        translations.append([*xy,float(plane['z_center']+size[2]/2)]);quaternions.append(rotation_to_quat(plane['rotation']))
        placements.append(dict(timestamp_us=int(frame['timestamp']),center_asset_m=translations[-1],yaw_world_rad=float(yaw),road_z_asset_m=plane['z_center'],footprint_flatness_m=plane['max_corner_residual_m']))
    token=hashlib.sha256(('native-added:'+track).encode()).hexdigest()[:24];key='native_added_'+token
    if any(name.rsplit('_',1)[-1]==token for name in asset):raise ValueError('Added render token collides with an existing asset')
    arrays=dict(means=means,scales=log_scales,quats=q,features_dc=data['dc'],features_rest=data['rest'],opacities=data['opacity'])
    state={f'gauss_params.{name}':torch.as_tensor(np.asarray(value,dtype=np.float32).copy()) for name,value in arrays.items()}
    state['instance_trans']=torch.tensor(translations,dtype=torch.float32);state['instance_quats']=torch.tensor(np.asarray(quaternions),dtype=torch.float32)
    asset[key]=dict(config=dict(type='RigidSubModel',sh_degree=3,scale_dim=3,fourier_features_dim=None,log_timestamps=torch.tensor([int(f['timestamp']) for f in frames],dtype=torch.int64),native_track_id=track,native_added=True),state_dict=state)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:torch.save(asset,stream)
    report=dict(status='merged',track_id=track,render_token=token,asset_key=key,asset_path=str(Path(request['asset_path']).resolve()),point_count=len(means),original_dimensions_m=data['dimensions'].tolist(),requested_dimensions_m=size.tolist(),axis_scale=scale.tolist(),scaled_geometry_dimensions_m=(means.max(0)-means.min(0)).tolist(),placements=placements,
        max_footprint_flatness_m=max(p['footprint_flatness_m'] for p in placements),source_checkpoint=str(source.resolve()),output_checkpoint=str(target.resolve()),
        scope='New checkpoint includes added rigid Gaussians, covariance scaling and road-grounded per-frame poses. Source checkpoint unmodified; rendering still required. PLY +X forward/+Z up is the existing asset convention.')
    Path(request['report_path']).write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--request',type=Path,required=True);args=parser.parse_args();merge(json.loads(args.request.read_text()))
if __name__=='__main__':main()
