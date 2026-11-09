"""Quick source-scene review with a direct, validated handoff to tool05."""
from contextvars import ContextVar
import json
import math
from pathlib import Path
from mcp_servers.scene_editing.adapter.schemas import AttackInput
from tools.llm_runtime import LLMTool, validation_details
from .schemas import Input, Decision, Output
from .observations import settings, sample_indices, Observation, ViewClient, ACTIVE_OBSERVATION

_INSPECTION=ContextVar('edit_planner_inspection',default=None)


def inspect_scene(data):
    cached=_INSPECTION.get()
    if cached is not None and cached[0]==data.model_dump(mode='json'):return cached[1]
    ref=data.scene_ref;missing=[]
    if data.retrieval_result.mode=='mock':missing.append('Mock retrieval does not establish a real source scene')
    frames_path=Path(ref.frames_path).expanduser()
    if not frames_path.is_file():
        missing.append('Missing scene annotations: '+str(frames_path));frames=[]
    else:
        frames=json.loads(frames_path.read_text())
        if not isinstance(frames,list) or not frames or any(not isinstance(f,dict) for f in frames):
            raise ValueError('Scene annotations must contain a nonempty frame-object list')
        if len(frames)!=ref.frame_count:raise ValueError('Annotation frame count differs from retrieved scene binding')
        times=[f['timestamp'] for f in frames]
        if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Frame timestamps must increase strictly')
    if not Path(ref.cameras_path).expanduser().is_file():missing.append('Missing scene camera calibration metadata')
    assets=list(dict.fromkeys(p for p in [ref.runtime_asset_path,ref.asset_path] if p and Path(p).expanduser().exists()))
    if not assets:missing.append('No available Gaussian scene asset')
    visual_settings=settings()
    indices=sample_indices(len(frames),visual_settings['time_samples']) if frames else []
    catalog={};eligible=None
    for frame in frames:
        anns=frame.get('anns',{})
        ids,names,boxes=[anns.get(key,[]) for key in ['track_tokens','gt_names','gt_boxes']]
        if not len(ids)==len(names)==len(boxes):raise ValueError('Track IDs, labels and boxes must align in every source frame')
        observed=set()
        for track,name,box in zip(ids,names,boxes):
            track=str(track)
            entry=catalog.setdefault(track,dict(track_id=track,label=name,observed_frames=0));entry['observed_frames']+=1
            valid=(isinstance(box,list) and len(box)>=7 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in box[:7]) and all(v>0 for v in box[3:6]))
            if name=='vehicle' and valid:observed.add(track)
        eligible=observed if eligible is None else eligible & observed
    eligible=eligible or set()
    for track,entry in catalog.items():entry['editable_existing']=track in eligible
    binding={};manifest=Path(ref.manifest_path).expanduser()
    if manifest.is_file():
        value=json.loads(manifest.read_text())
        if isinstance(value,dict) and 'timestamps_us' in value:
            if value.get('scene_token')!=ref.scene_token or value['timestamps_us']!=[f['timestamp'] for f in frames]:
                raise ValueError('Retrieval manifest belongs to a different scene or timeline')
            binding=value
    snapshots=[];images=[];image_evidence=[]
    camera_names=[visual_settings['initial_camera']]
    snapshot_indices={indices[0],indices[len(indices)//2],indices[-1]} if indices else set()
    for index in indices:
        frame=frames[index];anns=frame.get('anns',{})
        if index in snapshot_indices:
            objects=[]
            for i,track in enumerate(anns.get('track_tokens',[])):
                velocities=anns.get('gt_velocity_3d',[])
                objects.append(dict(track_id=str(track),label=anns['gt_names'][i],box=anns['gt_boxes'][i],velocity=velocities[i] if i<len(velocities) else None))
            snapshots.append(dict(frame_index=index,timestamp=frame['timestamp'],ego2global=frame.get('ego2global'),lidar2global=frame.get('lidar2global'),objects=objects))
        for camera_name in camera_names:
            camera=frame.get('cams',{}).get(camera_name,{})
            name=next((camera.get(k) for k in ['data_path','image_path','filename','image_file'] if camera.get(k)),None)
            possible=[];bound=binding.get('cameras',{}).get(camera_name,{}).get('frames',[])
            if index<len(bound) and bound[index].get('timestamp_us')==frame['timestamp'] and bound[index].get('image_path'):
                possible.append(Path(bound[index]['image_path']))
            if isinstance(name,str):
                path=Path(name).expanduser()
                possible += [path] if path.is_absolute() else [Path(ref.sample_root)/'original_images'/path,Path(ref.sample_root)/path]
            image=next((p for p in possible if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'}),None)
            if image is not None and image not in images:
                images.append(image);image_evidence.append(dict(frame_index=index,timestamp=frame['timestamp'],camera=camera_name,image_path=str(image)))
    if not images:missing.append('No source keyframes available for quick scene review')
    road=Path(ref.sample_root)/'metadata/map_drivable_official.json'
    # 地图缺失降级为提示，不再拦截 tool04；由 tool05 执行时检查
    vehicle=data.vehicle
    vehicle_available=bool(vehicle and Path(vehicle.asset_path).expanduser().is_file() and all(math.isfinite(v) for v in vehicle.size_m))
    retrieval=dict(video_path=data.retrieval_result.video_path,source_file=data.retrieval_result.source_file,source_segment=data.retrieval_result.source_segment)
    evidence=dict(annotation_path=str(frames_path),gaussian_assets=assets,frame_indices=indices,source_images=[str(p) for p in images],
        keyframes=image_evidence,visual_settings=visual_settings,eligible_vehicle_tracks=sorted(eligible),road_geometry_available=road.is_file(),
        missing_resources=missing,added_vehicle_asset_available=vehicle_available,retrieval=retrieval,
        review_scope='Whole bound scene; source_segment is only the retrieval match, not a synchronized crop')
    summary=dict(scene_id=ref.scene_id,scene_token=ref.scene_token,matched_camera=ref.matched_camera,frame_count=len(frames),
        actors=list(catalog.values()),snapshots=snapshots,evidence=evidence,
        box_coordinates='Original source annotation coordinates; transforms are supplied where available.',
        scene_requirement=data.scene_requirement,target_effect=data.target_effect,edit_strategy=data.edit_strategy)
    return summary,eligible,images


def payload(data):
    summary,_,_=inspect_scene(data)
    observation=Observation(data,summary)
    ACTIVE_OBSERVATION.set(observation)
    return observation.initial_content()


def enrich(data,response):
    summary,_,_=inspect_scene(data)
    result=dict(response,scene_id=data.scene_ref.scene_id,scene_token=data.scene_ref.scene_token,
        target_effect=data.target_effect,evidence=summary['evidence'],attack_input=None)
    if result['suitability']=='ready':
        plan=result['edit_plan']
        if plan['operation']=='add_vehicle' and not summary['evidence']['added_vehicle_asset_available']:
            result['suitability']='needs_information'
            result['reasons']=[*result['reasons'],'Provide a real added-vehicle PLY and length/width/height before tool05.']
        else:
            arguments=dict(scene_ref=data.retrieval_result_ref or data.scene_ref.model_dump(mode='json'),operation=plan['operation'],instruction=plan['instruction'])
            if plan['operation']=='edit_existing':arguments['target_track']=plan['target_track']
            else:arguments.update(maneuver=plan['maneuver'],vehicle=data.vehicle.model_dump(mode='json'))
            AttackInput.model_validate(arguments)
            result['attack_input']=arguments
    return result


def validate(data,result):
    summary,eligible,_=inspect_scene(data)
    if result.edit_plan is not None and result.edit_plan.operation=='edit_existing' and result.edit_plan.target_track not in eligible:
        raise ValueError('Existing target must be a valid vehicle observed in every source frame, as required by tool05')

TOOL=LLMTool(name='edit_planner',directory=Path(__file__).parent,input_model=Input,response_model=Decision,
    output_model=Output,payload_builder=payload,enrich_result=enrich,validate_result=validate,
    client_adapter=lambda client,data:ViewClient(client,ACTIVE_OBSERVATION.get()))


def run(input_data):
    try:data=Input.model_validate({k:v for k,v in input_data.items() if k!='_llm_client'})
    except (AttributeError,ValueError,TypeError,KeyError) as error:
        return dict(status='error',tool='edit_planner',error=dict(type='input_validation_error',message='Tool input does not match its schema',validation_errors=validation_details(error)))
    try:inspection=inspect_scene(data)
    except (OSError,ValueError,TypeError,KeyError) as error:
        return dict(status='error',tool='edit_planner',error=dict(type='scene_inspection_error',message='Cannot inspect the retrieved scene',validation_errors=validation_details(error)))
    token=_INSPECTION.set((data.model_dump(mode='json'),inspection))
    observation_token=ACTIVE_OBSERVATION.set(None)
    try:
        return TOOL.execute(input_data)
    finally:
        ACTIVE_OBSERVATION.reset(observation_token)
        _INSPECTION.reset(token)
