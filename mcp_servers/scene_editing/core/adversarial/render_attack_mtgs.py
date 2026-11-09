"""Render a recorded camera with only the target MTGS trajectory edited."""
from workflow_timing import timed_operation

import argparse, csv, json, math, shutil
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
from experiment_runtime import atomic_json
from render_ego_mtgs import (ROOT, asset_from_first_ego, camera_intrinsics, instantiate_models,
    load_checkpoint, matrix_from_rotation_translation, render_frame)
from run_ego_reference import DEFAULT_TARGET_TRACK, scene_from_frames

CAMERAS=('CAM_L0','CAM_F0','CAM_R0')

class TargetOverride:
    def __init__(self, model, displacement, yaw_delta, timestamp):
        self.model=model
        self.displacement=displacement
        self.yaw_delta=yaw_delta
        self.timestamp=timestamp
    def get_global_gaussians(self, timestamp):
        from worldengine.render.mtgs.utils.gaussian_utils import quat_mult
        q,t,_=self.model._get_log_pose_from_timestamp(timestamp)
        d=torch.tensor(self.displacement,dtype=t.dtype,device=t.device)
        h=self.yaw_delta/2
        r=torch.tensor([math.cos(h),0.,0.,math.sin(h)],dtype=q.dtype,device=q.device)
        new_q=quat_mult(r[None],q[None])[0]
        self.recorded_position=t.detach().cpu().numpy().copy()
        self.edited_position=(t+d).detach().cpu().numpy().copy()
        return self.model.get_global_gaussians(quat=new_q,trans=t+d,timestamp=timestamp)
    def get_gaussian_rgbs(self, **kwargs):
        return self.model.get_gaussian_rgbs(**kwargs)

def displacement_in_asset(candidate_xy, recorded_xy, asset_from_plan):
    delta=np.asarray(candidate_xy,float)-np.asarray(recorded_xy,float)
    return np.r_[asset_from_plan[:2,:2]@delta,0.]

def grounded_displacement_in_asset(candidate_xy, recorded_xy, transform, road_map,
                                    recorded_translation, box_height_m, diagnostics=None):
    """Apply the planar edit and place the recorded box center above its new road."""
    delta = displacement_in_asset(candidate_xy, recorded_xy, transform)
    translation = np.asarray(recorded_translation, float)
    if translation.shape != (3,) or not np.isfinite(translation).all() or box_height_m <= 0:
        raise ValueError('Invalid target translation or height')
    edited_xy = translation[:2] + delta[:2]
    height = road_map.height_with_local_plane(*edited_xy)
    if diagnostics is not None:
        diagnostics.update(height)
    delta[2] = height['height'] + box_height_m/2 - translation[2]
    return delta

def validate_replay_position(recorded_xy, annotated_xy, frame_index, limit_m=.5):
    residual = float(np.linalg.norm(np.asarray(recorded_xy)-np.asarray(annotated_xy)))
    if not np.isfinite(residual) or residual > limit_m:
        raise ValueError(f'Attack asset replay diverges {residual:.3f} m at frame {frame_index}; '
                         'choose an explicit supported --frames window')
    return residual

@timed_operation('render.attack', 'out')
def run(src,parent,out,worldengine,indices=None,target_track=DEFAULT_TARGET_TRACK):
    src=Path(src).resolve();parent=Path(parent).resolve();out=Path(out).resolve()
    if not out.is_relative_to((ROOT/'experiments').resolve()) or out.exists():raise ValueError('Output must be a new experiments directory')
    selected=json.loads((parent/'selected-candidate.json').read_text())['candidate_id']
    path=parent/'candidates'/selected/'trajectory.csv'; evaluation=json.loads((path.parent/'evaluation.json').read_text())
    if not evaluation['reference_screen_passed']:raise ValueError('Candidate not screened')
    frames=json.loads((src/'metadata/frames.json').read_text());times,ego,eyaw,objects=scene_from_frames(frames,target_track)
    rows=list(csv.DictReader(path.open()));
    if len(rows)!=len(times) or any(abs(float(r['time_s'])-times[i])>1e-6 for i,r in enumerate(rows)):raise ValueError('Time grid mismatch')
    chosen=list(range(len(times))) if indices is None else indices
    if not chosen or chosen!=sorted(set(chosen)) or min(chosen)<0 or max(chosen)>=len(times):raise ValueError('Invalid frame indices')
    checkpoint=list((src/'source_asset/background').glob('*.ckpt'))
    if len(checkpoint)!=1:raise ValueError('Expected one checkpoint')
    asset,globals_found=load_checkpoint(checkpoint[0]); recon=np.asarray(asset['background']['config']['recon2world_translation'],float)
    asset_from_plan=asset_from_first_ego(frames[0],recon); models=instantiate_models(asset,worldengine)
    target_key='rigid_object_'+target_track
    if target_key not in models:raise ValueError('Target target missing from MTGS asset')
    target_model=models[target_key]
    from road_height import RoadHeightMap, ego_height_reference
    road_map = RoadHeightMap.load(src)
    height_ref = ego_height_reference(frames, road_map, recon)
    out.mkdir(parents=True);(out/'input').mkdir();shutil.copyfile(path,out/'input/candidate-trajectory.csv')
    atomic_json(out/'config.json',dict(version='target_attack_mtgs.v1',input_dir=str(src),parent_run=str(parent),candidate_id=selected,checkpoint=str(checkpoint[0]),checkpoint_globals=globals_found,cameras=list(CAMERAS),target_track=target_track,frame_indices=chosen,actor_policy='recorded MTGS target pose plus candidate minus recorded annotation planar delta; original checkpoint residual preserved; other models unchanged',physical_execution='not_run'))
    writers={c:imageio.get_writer(out/f'new-camera-{c}.mp4',fps=2.,codec='libx264',macro_block_size=2) for c in CAMERAS}; entries=[];maxerr=0.
    try:
        for i in chosen:
            frame=frames[i];obj=next(x for x in objects[i] if x['track']==target_track); row=rows[i]
            xy=np.array([float(row['target_x_m']),float(row['target_y_m'])]);angle=float(row['target_yaw_rad']);delta=displacement_in_asset(xy,[obj['x'],obj['y']],asset_from_plan);yaw_delta=angle-obj['yaw']
            _, log_translation, _ = target_model._get_log_pose_from_timestamp(int(frame['timestamp']))
            target_index = list(map(str, frame['anns']['track_tokens'])).index(target_track)
            height = float(frame['anns']['gt_boxes'][target_index][5])
            height_diagnostic = {}
            delta = grounded_displacement_in_asset(xy, [obj['x'],obj['y']], asset_from_plan,
                road_map, log_translation.detach().cpu().numpy(), height, height_diagnostic)
            override=TargetOverride(target_model,delta,yaw_delta,int(frame['timestamp']));models[target_key]=override
            ego_matrix=np.asarray(frame['ego2global'],float).copy();ego_matrix[:3,3]-=recon
            ego_matrix[2,3] = road_map.height_at(*ego_matrix[:2,3]) + height_ref['reference_m']
            cameras={}
            for c in CAMERAS:
                info=frame['cams'][c]; image=src/'original_images'/info['data_path']
                with Image.open(image) as im:width,height=im.size
                intr,maps,width,height=camera_intrinsics(info,width,height)
                cam=ego_matrix@np.asarray(frame['lidar2ego'],float)@matrix_from_rotation_translation(info['sensor2lidar_rotation'],info['sensor2lidar_translation'])
                img,error=render_frame(models,cam,intr,maps,width,height,int(frame['timestamp']));maxerr=max(maxerr,error)
                if error>2e-4:raise ValueError('Camera readback failed')
                folder=out/'frames'/c;folder.mkdir(parents=True,exist_ok=True);Image.fromarray(img).save(folder/f'frame{i:03d}.png');writers[c].append_data(img)
                cameras[c]={'image_size':[width,height],'camera_readback_error':error}
            models[target_key]=target_model
            annotated=(asset_from_plan@np.array([obj['x'],obj['y'],0,1]))[:2]
            residual=validate_replay_position(override.recorded_position[:2], annotated, i)
            if np.linalg.norm((override.edited_position-override.recorded_position)-delta)>1e-4:raise ValueError('Target delta readback mismatch')
            entry=dict(frame_index=i,time_s=float(times[i]),timestamp=int(frame['timestamp']),candidate_xy=xy.tolist(),displacement_asset=delta.tolist(),recorded_asset_xy=override.recorded_position[:2].tolist(),edited_asset_xy=override.edited_position[:2].tolist(),recorded_model_vs_annotation_m=residual,target_ground_height=height_diagnostic,cameras=cameras)
            entries.append(entry);print(json.dumps(entry),flush=True)
    finally:
        for writer in writers.values():writer.close()
    decoded={}
    for c in CAMERAS:
        name=f'new-camera-{c}.mp4';reader=imageio.get_reader(out/name);meta=reader.get_meta_data();count=sum(1 for _ in reader);reader.close()
        if count!=len(chosen) or abs(meta['fps']-2.)>1e-6:raise ValueError('Video validation failed')
        decoded[name]=count
    report=dict(status='passed',full_window=chosen==list(range(len(times))),rendered_frame_ids=chosen,cameras=list(CAMERAS),decoded_videos=decoded,camera_readback_max_abs_error=maxerr,per_frame=entries,rendered_frames=len(chosen),max_recorded_model_vs_annotation_m=max(e['recorded_model_vs_annotation_m'] for e in entries),scope='Reference target edit; model versus annotation residual retained; no CARLA or physical collision')
    atomic_json(out/'render-report.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--parent-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worldengine',type=Path,required=True);p.add_argument('--frames',type=int,nargs='+');p.add_argument('--target-track',required=True);a=p.parse_args();run(a.input,a.parent_run,a.output,a.worldengine,a.frames,a.target_track)
