"""Export the exact edited frames/expert poses to a native WE trajectory package.

Uses real calibrated cameras, annotations and the scene's MTGS checkpoint. Missing
assets produce an explicit unavailable report, never a substitute original scene.
"""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import pickle
import os
import subprocess
import numpy as np
from experiment_runtime import atomic_json
from shared_history import (
    MAX_NATIVE_GRID_JITTER_S, WE_TICK_S, native_sample_rate, shared_history_end_index,
)
from run_ego_reference import ego_path_from_frames,observed_objects_from_frames,EGO_FRONT,EGO_REAR,EGO_WIDTH


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def native_types(scene):
    """Match the official scenario pkl contract: calibration/pose fields are
    numpy arrays, not JSON lists (MTGS get_transform_matrix and the navformer
    calibration parser call ndarray/tensor methods on them). The field set
    mirrors the official navtrain pkls; lidar2ego_rotation stays a list there,
    so it stays a list here too."""
    frame_fields=('lidar2ego','lidar2global','ego2global','can_bus','driving_command',
        'ego2global_translation','ego2global_rotation','lidar2ego_translation')
    cam_fields=('sensor2lidar_rotation','sensor2lidar_translation','sensor2ego_rotation',
        'sensor2ego_translation','cam_intrinsic','intrinsic')
    converted=[0]
    def arrays(mapping,fields):
        for key in fields:
            value=mapping.get(key)
            if isinstance(value,list):
                mapping[key]=np.asarray(value)
                assert np.array_equal(np.asarray(value),mapping[key]);converted[0]+=1
    arrays(scene['lidar'],('sensor2ego_rotation','sensor2ego_translation'))
    for camera in scene['cameras'].values():arrays(camera,('sensor2ego_rotation','sensor2ego_translation'))
    for frame in scene['metadata']['openscene_data_infos_dict'].values():
        arrays(frame,frame_fields)
        for camera in frame.get('cams',{}).values():arrays(camera,cam_fields)
    ego2globals=scene['metadata'].get('digitaltwin_ego2globals')
    if isinstance(ego2globals,list) and ego2globals and isinstance(ego2globals[0],list):
        scene['metadata']['digitaltwin_ego2globals']=[np.asarray(m) for m in ego2globals]
        converted[0]+=len(ego2globals)
    return converted[0]


def export_summary(report):
    alignment=report.get('checks', {}).get('time_alignment')
    if alignment:
        alignment={key:value for key,value in alignment.items() if key not in ('native_time_s','uniform_time_s')}
        alignment['max_grid_drift_s']=report['checks'].get('max_uniform_grid_drift_s')
    return {key: report[key] for key in ('status', 'reason', 'error_type', 'report_path') if key in report} | {
        'time_alignment': alignment}


def validate_expert_trajectory(times, recorded_xy, recorded_yaw, expert):
    """Validate the submitted poses before native export or rendering."""
    et, xy, yaw = (np.asarray(component) for component in expert)
    if et.shape != times.shape or not np.allclose(et, times, atol=1e-6, rtol=0):
        raise ValueError('Expert timestamps differ from frozen attack')
    if xy.shape != recorded_xy.shape or yaw.shape != recorded_yaw.shape:
        raise ValueError('Expert pose count differs from frozen attack')
    if not np.isfinite(xy).all() or not np.isfinite(yaw).all():
        raise ValueError('Expert poses must be finite')

    prefix = slice(0, shared_history_end_index(times) + 1)
    yaw_delta = yaw[prefix] - recorded_yaw[prefix]
    yaw_error = np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)))
    if not np.allclose(xy[0], recorded_xy[0], atol=1e-6, rtol=0) or yaw_error[0] > 1e-6:
        raise ValueError('Expert changed the initial scene pose')
    if not np.allclose(xy[prefix], recorded_xy[prefix], atol=1e-6, rtol=0) or yaw_error.max() > 1e-6:
        raise ValueError('Expert changed the shared 0–1.5s history; edit only after takeover')


def build(frames,cameras,scene_id,asset_id,*,expert=None,reconstruction=None):
    """WE states use body centres in world-axis coordinates translated by the first rear axle."""
    times,recorded_xy,recorded_yaw=ego_path_from_frames(frames)
    times=np.asarray(times);first=np.asarray(frames[0]['ego2global'],float)
    yaw0=float(np.arctan2(first[1,0],first[0,0]));rotation=np.array([[np.cos(yaw0),-np.sin(yaw0)],[np.sin(yaw0),np.cos(yaw0)]])
    xy,yaw=recorded_xy,recorded_yaw
    if expert is not None:
        validate_expert_trajectory(times, recorded_xy, recorded_yaw, expert)
        _, xy, yaw = expert
    # Align acquisition jitter by native frame index, preserving images and every
    # actor pose together. Large irregularity/missing frames still require resampling.
    sample_rate = native_sample_rate(times)
    if sample_rate<1:raise ValueError('Scene sampling is faster than WE native base interval')
    nominal_step = sample_rate * WE_TICK_S
    uniform = np.arange(len(times)) * nominal_step
    drift = float(np.max(np.abs(uniform - times)))
    tolerance=min(MAX_NATIVE_GRID_JITTER_S,nominal_step*.01)
    interval_drift=float(np.max(abs(np.diff(times)-nominal_step)))
    if drift>tolerance or interval_drift>tolerance:
        raise ValueError(f'Nonuniform timestamps need explicit resampling before native export: grid drift={drift:.6f}s, interval drift={interval_drift:.6f}s, alignment limit={tolerance:.6f}s')
    if reconstruction is None:
        raise ValueError('MTGS reconstruction calibration is required; raw UTM ego poses are not render references')
    actors=observed_objects_from_frames(frames);tracks={};n=len(times)
    types={'vehicle':'VEHICLE','pedestrian':'PEDESTRIAN','bicycle':'CYCLIST','cyclist':'CYCLIST','barrier':'TRAFFIC_BARRIER','traffic_cone':'TRAFFIC_CONE','cone':'TRAFFIC_CONE','generic_object':'TRAFFIC_OBJECT','traffic_object':'TRAFFIC_OBJECT','czone_sign':'TRAFFIC_OBJECT'}
    def track(token,kind):
        return dict(type=kind,metadata=dict(type=kind,track_length=n,object_id=token,nuplan_id=token),
            state=dict(position=np.zeros((n,3)),heading=np.zeros(n),velocity=np.zeros((n,2)),valid=np.zeros(n),
                length=np.zeros((n,1)),width=np.zeros((n,1)),height=np.zeros((n,1))))
    for i,objects in enumerate(actors):
        for actor in objects:
            token=actor['track'];kind=types.get(actor['label'].lower())
            if kind is None:raise ValueError('Unsupported annotated object type: '+actor['label'])
            node=tracks.setdefault(token,track(token,kind));s=node['state'];s['position'][i,:2]=np.array([actor['x'],actor['y']])@rotation.T
            s['heading'][i]=actor['yaw']+yaw0;s['valid'][i]=1;s['length'][i]=actor['length'];s['width'][i]=actor['width']
            j=list(map(str,frames[i]['anns']['track_tokens'])).index(token);s['height'][i]=frames[i]['anns']['gt_boxes'][j][5]
    ego=track('ego','VEHICLE');s=ego['state'];heading=np.asarray(yaw)+yaw0
    offset=(EGO_FRONT-EGO_REAR)/2
    s['position'][:,:2]=np.asarray(xy)@rotation.T+offset*np.column_stack([np.cos(heading),np.sin(heading)])
    s['heading'][:]=heading;s['valid'][:]=1;s['length'][:]=EGO_FRONT+EGO_REAR;s['width'][:]=EGO_WIDTH;s['height'][:]=1.777
    tracks['ego']=ego
    for node in tracks.values():
        state=node['state'];valid=state['valid'].astype(bool);indices=np.flatnonzero(valid)
        if len(indices)>1:state['velocity'][valid]=np.gradient(state['position'][valid,:2],uniform[valid],axis=0)
    # Shared history has causal velocities. Expert future changes must not leak
    # through a central difference into the VAD takeover state.
    shared_end = shared_history_end_index(times)
    for i in range(shared_end + 1):
        left=max(0,i-1);right=max(1,i)
        ego['state']['velocity'][i]=(ego['state']['position'][right,:2]-ego['state']['position'][left,:2])/(uniform[right]-uniform[left])
    # Preserve original sensor/annotation data; expert poses change sensor frame,
    # so reproject each box/velocity instead of leaving attack annotations stale.
    exported=deepcopy(frames)
    for i,f in enumerate(exported):
        f['timestamp']=frames[0]['timestamp']+int(round(uniform[i]*1_000_000))
        old=np.asarray(frames[i]['ego2global'],float);new=old.copy();delta=float(yaw[i]-recorded_yaw[i]);cs,sn=np.cos(delta),np.sin(delta)
        rz=np.array([[cs,-sn,0.],[sn,cs,0.],[0.,0.,1.]])
        new[:3,:3]=rz@old[:3,:3];new[:2,3]=first[:2,3]+np.asarray(xy)[i]@rotation.T
        f['ego2global']=new.tolist();f['ego2global_translation']=new[:3,3].tolist()
        # Source quaternion is wxyz; left multiply by the planar yaw quaternion.
        q=np.asarray(frames[i].get('ego2global_rotation',[1.,0.,0.,0.]),float);w,z=np.cos(delta/2),np.sin(delta/2)
        qnew=np.array([w*q[0]-z*q[3],w*q[1]-z*q[2],w*q[2]+z*q[1],w*q[3]+z*q[0]])
        f['ego2global_rotation']=qnew.tolist()
        old_lidar=old@np.asarray(f['lidar2ego'],float);new_lidar=new@np.asarray(f['lidar2ego'],float);change=np.linalg.inv(new_lidar)@old_lidar
        angle=float(np.arctan2(change[1,0],change[0,0]))
        for j,box in enumerate(f['anns']['gt_boxes']):
            b=np.asarray(box,float);b[:3]=(change@np.r_[b[:3],1.])[:3];b[6]+=angle;f['anns']['gt_boxes'][j]=b.tolist()
        if 'gt_velocity_3d' in f['anns']:
            world_velocity=np.array([np.r_[tracks[str(token)]['state']['velocity'][i],0.]
                for token in f['anns']['track_tokens']]).reshape(-1,3)
            f['anns']['gt_velocity_3d']=(world_velocity@new_lidar[:3,:3]).tolist()
        f['lidar2global']=new_lidar.tolist()
        if 'can_bus' in f:
            f['can_bus'][:3]=new[:3,3].tolist();f['can_bus'][3:7]=qnew.tolist()
    # Rendering reference belongs to the reconstruction, not UTM annotations.
    metadata=dict(nuplan_lidar_pc_tokens=[f['token'] for f in frames],
        openscene_data_infos_dict={f['token']:f for f in exported},digitaltwin_asset_id=asset_id,
        digitaltwin_ego2globals=reconstruction.poses.copy(),
        old_origin_in_current_coordinate=-first[:2,3],native_export=dict(version='edited_scene.v2',
            original_timestamps_us=[f['timestamp'] for f in frames],max_uniform_grid_drift_s=drift,
            time_alignment=dict(method='native_index_jitter_alignment.v1',native_time_s=times.tolist(),
                uniform_time_s=uniform.tolist(),max_interval_drift_s=interval_drift,
                tolerance_s=tolerance,poses_and_sensor_frames_preserved=True,
                shared_history_end_frame_index=shared_end,
                shared_history_source_end_s=float(times[shared_end])),
            ego_position_reference='body_center; rear axle offset '+str(offset)+'m',
            ego_calibration_frame='reconstruction',reconstruction_calibration=reconstruction.provenance,
            planar_geometry=True,physical_execution_certified=False,
            recorded_ego_time_s=uniform.tolist(),
            recorded_ego_rear_axle_global=(first[:2,3]+recorded_xy@rotation.T).tolist(),
            recorded_ego_heading_rad=(recorded_yaw+yaw0).tolist()))
    nav_heading=recorded_yaw+yaw0
    metadata['native_export']['navigation_reference_body_xy_m']=(recorded_xy@rotation.T+
        offset*np.column_stack([np.cos(nav_heading),np.sin(nav_heading)])).tolist()
    scene=dict(id=scene_id,name=scene_id,token=frames[0].get('scene_token',scene_id),dataset='digitaltwin.nuplan',
        map=frames[0]['map_location'],log_length=n,sample_rate=sample_rate,base_timestamp=frames[0]['timestamp'],
        sdc_id='ego',cameras=deepcopy(cameras),lidar=dict(channel='LIDAR_TOP',sensor2ego_rotation=frames[0]['lidar2ego_rotation'],sensor2ego_translation=frames[0]['lidar2ego_translation']),
        object_track=tracks,metadata=metadata,dynamic_map_states={},map_features={})
    # Signal states/positions are attached after the real lane-map binding.
    typed_fields=native_types(scene)
    return scene,dict(max_uniform_grid_drift_s=drift,time_alignment=metadata['native_export']['time_alignment'],typed_calibration_fields=typed_fields,original_ego_preserved=expert is None,
        frame_count=n,tracks=list(tracks),reconstruction_calibration=reconstruction.provenance)


def export(source_ref,frozen_scene,output,*,expert_trajectory=None,attack_result=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=Path(source_ref['sample_root']);frozen=Path(frozen_scene)
    frames_path=frozen/'metadata/frames.json';frames=json.loads(frames_path.read_text())
    report=dict(version='native_export.v1',status='unavailable',simulation_input=None,source_frames=str(frames_path),source_frames_sha256=digest(frames_path))
    try:
        camera_path=Path(source_ref.get('cameras_path') or source/'metadata/cameras.json')
        cameras=json.loads(camera_path.read_text())
        if not cameras:raise ValueError('No real camera calibrations available')
        for name,camera in cameras.items():
            if not {'sensor2ego_rotation','sensor2ego_translation','intrinsic','distortion','height','width'}<=set(camera):raise ValueError('Incomplete camera calibration: '+name)
        checkpoints=list((source/'source_asset/background').glob('*.ckpt'))
        if len(checkpoints)!=1:raise ValueError('Require exactly one real scene MTGS checkpoint')
        checkpoint=checkpoints[0];asset_id=checkpoint.stem
        added=None
        if attack_result and attack_result.get('operation')=='add_vehicle':
            info=json.loads((frozen/'freeze-info.json').read_text())
            from ego_history import source_scene
            if source_scene(info['source_scene'])!=source.resolve():raise ValueError('Added asset provenance references a different source scene')
            track=attack_result.get('target_track',info.get('track_id'))
            if info.get('track_id')!=track or not info.get('asset_path'):raise ValueError('Added vehicle needs its exact frozen track and PLY asset binding')
            original=json.loads((source/'metadata/frames.json').read_text())
            if any(track in map(str,f['anns']['track_tokens']) for f in original):raise ValueError('Added vehicle track already exists in original observations')
            added=dict(info,track_id=track)
        expert=None
        if expert_trajectory:
            with Path(expert_trajectory).open() as stream:rows=list(csv.DictReader(stream))
            expert=(np.array([float(r['time_s']) for r in rows]),np.array([[float(r['ego_x_m']),float(r['ego_y_m'])] for r in rows]),np.array([float(r['ego_yaw_rad']) for r in rows]))
        from reconstruction_calibration import load_reconstruction_calibration
        reconstruction=load_reconstruction_calibration(source,frames,asset_id=asset_id)
        scene,checks=build(frames,cameras,source_ref['scene_id'],asset_id,expert=expert,reconstruction=reconstruction)
        history=frozen/'ego-history.json'
        if history.is_file():
            data=json.loads(history.read_text());poses=np.asarray(data['original_navigation_ego2global'],float)
            angles=np.arctan2(poses[:,1,0],poses[:,0,0]);offset=(EGO_FRONT-EGO_REAR)/2
            scene['metadata']['native_export']['navigation_reference_body_xy_m']=(poses[:,:2,3]-
                np.asarray(frames[0]['ego2global'])[:2,3]+offset*np.column_stack([np.cos(angles),np.sin(angles)])).tolist()
            nav=np.asarray(scene['metadata']['native_export']['navigation_reference_body_xy_m'])
            interval=float(scene['sample_rate'])*.05
            scene['object_track']['ego']['state']['velocity'][0]=(nav[1]-nav[0])/interval
            scene['metadata']['native_export']['shared_history']=dict(end_s=1.5,
                targets=data['targets'],source_frames_sha256=data['source_frames_sha256'])
        # Known official-map coordinates have an explicit rear-axle/yaw origin.
        official=source/'metadata/map_drivable_official.json'
        if not official.is_file():raise ValueError('Official map with explicit coordinate provenance is required')
        road=json.loads(official.read_text())
        if not road.get('frame_convention','').startswith('trajectory-local:'):raise ValueError('Unknown official map coordinate convention')
        # 数据集把这两项存成字符串；显式转 float，否则 np.cos 会抛 TypeError
        yaw0=float(road['first_frame_yaw_rad']);rot=np.array([[np.cos(yaw0),-np.sin(yaw0)],[np.sin(yaw0),np.cos(yaw0)]])
        origin=np.asarray(road['first_frame_rear_axle_global'],dtype=float)
        first=np.asarray(frames[0]['ego2global'],dtype=float)[:2,3]
        if not np.allclose(origin,first,atol=1e-6,rtol=0):raise ValueError('Official map origin differs from edited scene')
        from native_scene_map import align
        full_map_path=source/'metadata/map_features.json'
        scene['map_features'],map_binding=align(json.loads(full_map_path.read_text()),road)
        checks['map_coordinate_binding']=map_binding
        scene['metadata']['native_export']['signal_source']='Source red flags: true=red, false=unknown; richer signal colors were not supplied'
        for i,frame in enumerate(frames):
            for lane,red in frame.get('traffic_lights',[]):
                token=str(lane);geometry=scene['map_features'].get(token)
                if geometry is None or 'polyline' not in geometry:
                    raise ValueError('Observed signal has no source lane centreline: '+token)
                points=np.asarray(geometry['polyline'])[:,:2]
                arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
                position=[float(np.interp(min(8.,arc[-1]),arc,points[:,axis])) for axis in range(2)]
                node=scene['dynamic_map_states'].setdefault(token,dict(type='TRAFFIC_LIGHT',metadata=dict(type='TRAFFIC_LIGHT',track_length=len(frames),object_id=token,lane_id=token,dataset='nuplan'),
                    traffic_light_position=position,traffic_light_lane=token,state=dict(traffic_light_state=['TRAFFIC_LIGHT_UNKNOWN']*len(frames))))
                node['state']['traffic_light_state'][i]='TRAFFIC_LIGHT_RED' if red else 'TRAFFIC_LIGHT_UNKNOWN'
        assets=output/'assets';background=assets/asset_id/'background';background.mkdir(parents=True)
        native_checkpoint=background/checkpoint.name
        added_report=None
        if added is None:
            native_checkpoint.symlink_to(checkpoint.resolve())
        else:
            from backend_configuration import load_settings
            settings=load_settings()['render']
            request=dict(source_checkpoint=str(checkpoint.resolve()),output_checkpoint=str(native_checkpoint.resolve()),
                asset_path=str(Path(added['asset_path']).resolve()),size_m=added['size_m'],track_id=added['track_id'],
                source_scene=str(source.resolve()),frames_path=str(frames_path.resolve()),report_path=str((output/'added-asset-report.json').resolve()))
            request_path=output/'added-asset-request.json';atomic_json(request_path,request)
            env=dict(os.environ);env['LD_LIBRARY_PATH']=settings['system_lib']+os.pathsep+env.get('LD_LIBRARY_PATH','')
            with (output/'added-asset-merge.log').open('x') as stream:
                completed=subprocess.run([settings['python'],'-B',str(Path(__file__).with_name('native_vehicle_asset.py')),'--request',str(request_path.resolve())],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
            if completed.returncode:raise ValueError('Added Gaussian merge failed; inspect '+str(output/'added-asset-merge.log'))
            added_report=json.loads((output/'added-asset-report.json').read_text())
            if added_report.get('status')!='merged':raise ValueError('Added Gaussian asset did not produce a complete merge report')
            scene['metadata']['native_export']['added_vehicle_assets']=[dict(track_id=added['track_id'],render_token=added_report['render_token'])]
            checks['added_vehicle_asset']=dict(status='merged',track_id=added['track_id'],asset_point_count=added_report['point_count'],max_footprint_flatness_m=added_report['max_footprint_flatness_m'])
        scenario=output/'all_scenarios.pkl'
        with scenario.open('xb') as stream:pickle.dump({source_ref['scene_id']:scene},stream,protocol=4)
        # The rollout horizon is a property of this scene; exporting it removes any
        # chance that a downstream runner re-invents a different frame budget.
        package=dict(scenario_data_folder_path=str(output.resolve()),asset_folder_path=str(assets.resolve()),scenario_pkl_file_name=scenario.name,frame_count=len(frames))
        report.update(status='exported',scene_id=source_ref['scene_id'],simulation_input=package,checks=checks,
            scenario_sha256=digest(scenario),asset_checkpoint=str(native_checkpoint.resolve()),asset_checkpoint_sha256=digest(native_checkpoint),source_checkpoint_sha256=digest(checkpoint),
            added_asset_report_path=str(output/'added-asset-report.json') if added_report else None,added_ply_sha256=digest(added['asset_path']) if added else None,camera_calibration_sha256=digest(camera_path),full_map_sha256=digest(full_map_path),
            expert_trajectory_path=str(expert_trajectory) if expert_trajectory else None,
            expert_trajectory_sha256=digest(expert_trajectory) if expert_trajectory else None,
            scope='Exact native-index edited planar poses, real calibrated cameras and source checkpoint. Native replay/video decoding and physical acceptance have not been run.')
    except (OSError,ValueError,KeyError,TypeError,subprocess.TimeoutExpired) as error:
        report['reason']=str(error)
        report['error_type']=type(error).__name__
    report['report_path']=str((output/'export-report.json').resolve())
    atomic_json(output/'export-report.json',report)
    return report
