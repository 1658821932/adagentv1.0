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
from run_ego_reference import ego_path_from_frames,observed_objects_from_frames,EGO_FRONT,EGO_REAR,EGO_WIDTH


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def build(frames,cameras,scene_id,asset_id,*,expert=None):
    """WE states use body centres in world-axis coordinates translated by the first rear axle."""
    times,recorded_xy,recorded_yaw=ego_path_from_frames(frames)
    times=np.asarray(times);first=np.asarray(frames[0]['ego2global'],float)
    yaw0=float(np.arctan2(first[1,0],first[0,0]));rotation=np.array([[np.cos(yaw0),-np.sin(yaw0)],[np.sin(yaw0),np.cos(yaw0)]])
    xy,yaw=recorded_xy,recorded_yaw
    if expert is not None:
        et,xy,yaw=expert
        if not np.allclose(et,times,atol=1e-6,rtol=0):raise ValueError('Expert timestamps differ from frozen attack')
        if not np.isfinite(xy).all() or not np.isfinite(yaw).all():raise ValueError('Expert poses must be finite')
        if not np.array_equal(np.asarray(xy)[times<=3],recorded_xy[times<=3]) or not np.array_equal(np.asarray(yaw)[times<=3],recorded_yaw[times<=3]):
            raise ValueError('Expert changed the immutable first three seconds')
    # WE requires a uniform sample interval. Keep native poses at native indices,
    # retain original timestamps in metadata and report the small grid drift.
    step=float(np.median(np.diff(times)));sample_rate=round(step/.05)
    if sample_rate<1:raise ValueError('Scene sampling is faster than WE native base interval')
    uniform=np.arange(len(times))*sample_rate*.05;drift=float(np.max(np.abs(uniform-times)))
    if drift>.001:raise ValueError('Nonuniform timestamps need explicit resampling before native export')
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
        if len(indices)>1:state['velocity'][valid]=np.gradient(state['position'][valid,:2],times[valid],axis=0)
    # Preserve original sensor/annotation data; expert poses change sensor frame,
    # so reproject each box/velocity instead of leaving attack annotations stale.
    exported=deepcopy(frames)
    for i,f in enumerate(exported):
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
        if 'gt_velocity_3d' in f['anns']:f['anns']['gt_velocity_3d']=(np.asarray(f['anns']['gt_velocity_3d'])@change[:3,:3].T).tolist()
        f['lidar2global']=new_lidar.tolist()
        if 'can_bus' in f:
            f['can_bus'][:3]=new[:3,3].tolist();f['can_bus'][3:7]=qnew.tolist()
    # Camera and reconstruction calibration use the original poses, not edited ego.
    metadata=dict(nuplan_lidar_pc_tokens=[f['token'] for f in frames],
        openscene_data_infos_dict={f['token']:f for f in exported},digitaltwin_asset_id=asset_id,
        digitaltwin_ego2globals=[f['ego2global'] for f in frames],
        old_origin_in_current_coordinate=-first[:2,3],native_export=dict(version='edited_scene.v1',
            original_timestamps_us=[f['timestamp'] for f in frames],max_uniform_grid_drift_s=drift,
            ego_position_reference='body_center; rear axle offset '+str(offset)+'m',
            ego_calibration_frame='global',planar_geometry=True,physical_execution_certified=False))
    scene=dict(id=scene_id,name=scene_id,token=frames[0].get('scene_token',scene_id),dataset='digitaltwin.nuplan',
        map=frames[0]['map_location'],log_length=n,sample_rate=sample_rate,base_timestamp=frames[0]['timestamp'],
        sdc_id='ego',cameras=deepcopy(cameras),lidar=dict(channel='LIDAR_TOP',sensor2ego_rotation=frames[0]['lidar2ego_rotation'],sensor2ego_translation=frames[0]['lidar2ego_translation']),
        object_track=tracks,metadata=metadata,dynamic_map_states={},map_features={})
    # Signal states/positions are attached after the real lane-map binding.
    return scene,dict(max_uniform_grid_drift_s=drift,original_ego_preserved=expert is None,
        frame_count=n,tracks=list(tracks),source_ego2globals_preserved_for_reconstruction=True)


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
            if Path(info['source_scene']).resolve()!=source.resolve():raise ValueError('Added asset provenance references a different source scene')
            track=attack_result.get('target_track',info.get('track_id'))
            if info.get('track_id')!=track or not info.get('asset_path'):raise ValueError('Added vehicle needs its exact frozen track and PLY asset binding')
            original=json.loads((source/'metadata/frames.json').read_text())
            if any(track in map(str,f['anns']['track_tokens']) for f in original):raise ValueError('Added vehicle track already exists in original observations')
            added=dict(info,track_id=track)
        expert=None
        if expert_trajectory:
            with Path(expert_trajectory).open() as stream:rows=list(csv.DictReader(stream))
            expert=(np.array([float(r['time_s']) for r in rows]),np.array([[float(r['ego_x_m']),float(r['ego_y_m'])] for r in rows]),np.array([float(r['ego_yaw_rad']) for r in rows]))
        scene,checks=build(frames,cameras,source_ref['scene_id'],asset_id,expert=expert)
        # Known official-map coordinates have an explicit rear-axle/yaw origin.
        official=source/'metadata/map_drivable_official.json'
        if not official.is_file():raise ValueError('Official map with explicit coordinate provenance is required')
        road=json.loads(official.read_text())
        if not road.get('frame_convention','').startswith('trajectory-local:'):raise ValueError('Unknown official map coordinate convention')
        yaw0=road['first_frame_yaw_rad'];rot=np.array([[np.cos(yaw0),-np.sin(yaw0)],[np.sin(yaw0),np.cos(yaw0)]])
        origin=np.asarray(road['first_frame_rear_axle_global']);first=np.asarray(frames[0]['ego2global'])[:2,3]
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
            settings=json.loads((Path(__file__).with_name('backend_settings.json')).read_text())['render']
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
        package=dict(scenario_data_folder_path=str(output.resolve()),asset_folder_path=str(assets.resolve()),scenario_pkl_file_name=scenario.name)
        report.update(status='exported',scene_id=source_ref['scene_id'],simulation_input=package,checks=checks,
            scenario_sha256=digest(scenario),asset_checkpoint=str(native_checkpoint.resolve()),asset_checkpoint_sha256=digest(native_checkpoint),source_checkpoint_sha256=digest(checkpoint),
            added_asset_report_path=str(output/'added-asset-report.json') if added_report else None,added_ply_sha256=digest(added['asset_path']) if added else None,camera_calibration_sha256=digest(camera_path),full_map_sha256=digest(full_map_path),
            expert_trajectory_path=str(expert_trajectory) if expert_trajectory else None,
            expert_trajectory_sha256=digest(expert_trajectory) if expert_trajectory else None,
            scope='Exact native-index edited planar poses, real calibrated cameras and source checkpoint. Native replay/video decoding and physical acceptance have not been run.')
    except (OSError,ValueError,KeyError,TypeError,subprocess.TimeoutExpired) as error:
        report['reason']=str(error)
    atomic_json(output/'export-report.json',report)
    return report
