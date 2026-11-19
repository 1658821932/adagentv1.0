"""Quick source-scene review with a direct, validated handoff to tool05."""
from contextvars import ContextVar
import hashlib
import json
import math
from pathlib import Path

# Distance is a planning hint, not an editability requirement.
ELIGIBLE_MAX_DISTANCE_M = 30.0
ELIGIBLE_BEHIND_TOLERANCE_M = 10.0
from mcp_servers.scene_editing.adapter.schemas import AttackInput
from tools.llm_runtime import LLMTool, validation_details
from .schemas import Input, Decision, Output, EgoHistoryPlan
from .observations import settings, sample_indices, Observation, ViewClient, ACTIVE_OBSERVATION
from . import observations

_INSPECTION=ContextVar('edit_planner_inspection',default=None)
_CACHE_VERSION='asset_supplement.v3'


def _plan_cache_path(data):
    request=data.model_dump(mode='json');request.pop('vehicle',None)
    evidence=dict(request=request,version=_CACHE_VERSION,
        prompt=hashlib.sha256(Path(__file__).with_name('prompt.md').read_bytes()).hexdigest())
    for name,path in [('frames',data.scene_ref.frames_path),('manifest',data.scene_ref.manifest_path),
            ('cameras',data.scene_ref.cameras_path)]:
        file=Path(path)
        evidence[name]=hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
    key=hashlib.sha256(json.dumps(evidence,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    return observations.LOG_ROOT/'plan-cache'/f'{key}.json'


def _awaits_added_asset(response):
    if response.get('suitability')!='needs_information' or response.get('attack_input') is not None:
        return False
    options=response.get('edit_options') or []
    return ((bool(options) and all(option['operation']=='add_vehicle' for option in options))
            or (response.get('edit_plan') or {}).get('operation')=='add_vehicle')


def _cached_asset_supplement(data):
    if data.vehicle is None:
        return None
    path=_plan_cache_path(data)
    if not path.is_file():
        return None
    try:
        cached=json.loads(path.read_text())
        response=cached['response']
        if cached.get('version')!=_CACHE_VERSION or not _awaits_added_asset(response):
            return None
        result=enrich(data,dict(response,suitability='ready'))
        Output.model_validate(result)
        return dict(status='success',tool='edit_planner',result=result,
            meta=dict(mode='real',model='asset_supplement_cache',attempts=0,reused=True,elapsed_seconds=0.0))
    except (OSError,ValueError,TypeError,KeyError):
        return None


def _save_asset_supplement(data,result):
    response=result.get('result') or {}
    if data.vehicle is not None or not _awaits_added_asset(response):
        return
    path=_plan_cache_path(data);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(version=_CACHE_VERSION,response=response),ensure_ascii=False,indent=2))
    temporary.replace(path)


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
    catalog={};eligible=None;positions={}
    for frame in frames:
        anns=frame.get('anns',{})
        ids,names,boxes=[anns.get(key,[]) for key in ['track_tokens','gt_names','gt_boxes']]
        if not len(ids)==len(names)==len(boxes):raise ValueError('Track IDs, labels and boxes must align in every source frame')
        observed=set()
        for track,name,box in zip(ids,names,boxes):
            track=str(track)
            entry=catalog.setdefault(track,dict(track_id=track,label=name,observed_frames=0));entry['observed_frames']+=1
            valid=(isinstance(box,list) and len(box)>=7 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in box[:7]) and all(v>0 for v in box[3:6]))
            if name=='vehicle' and valid:
                observed.add(track)
                # Boxes are scene-local per frame, so this is the distance to the ego.
                positions.setdefault(track,[]).append((float(box[0]),float(box[1])))
        eligible=observed if eligible is None else eligible & observed
    eligible=eligible or set()
    # Preserve every fully observed valid vehicle for tool05 compatibility.
    # Proximity to the recorded ego is advisory: a useful edited conflict may
    # require a different motion than the original actor trajectory.
    for track in eligible:
        points=positions.get(track) or []
        distances=[math.hypot(x,y) for x,y in points]
        catalog[track]['nearest_ego_distance_m']=round(min(distances),1) if distances else None
        if any(d<=ELIGIBLE_MAX_DISTANCE_M and x>-ELIGIBLE_BEHIND_TOLERANCE_M for (x,y),d in zip(points,distances)):
            catalog[track]['edit_relevance']='near_recorded_ego'
        else:
            catalog[track]['edit_relevance']='far_or_behind_in_recording_only'
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
        scene_requirement=data.scene_requirement,target_effect=data.target_effect,edit_strategy=data.edit_strategy,
        ego_input_edit_strategy=data.ego_input_edit_strategy,
        editing_capabilities=dict(
            edit_existing='Edit a fully observed vehicle from its recorded initial state by default. Tool05 can create a bounded initial-position/yaw-adjusted version and must regenerate or rigidly transform the trajectory, then rerun every check.',
            add_vehicle='With a supplied asset, choose a new initial pose and speed; stationary can keep the new vehicle at that pose for the whole scene.',
            motion_feasibility='Detailed generated-trajectory checks belong to tool05. Original future motion is not a hard rejection in tool04.',
            distance_hints='Recorded proximity is advisory, not a vehicle eligibility or scenario rejection gate.'))
    return summary,eligible,images


def payload(data):
    summary,_,_=inspect_scene(data)
    observation=Observation(data,summary)
    ACTIVE_OBSERVATION.set(observation)
    return observation.initial_content()


def enrich(data,response):
    summary,eligible,_=inspect_scene(data)
    result=dict(response,scene_id=data.scene_ref.scene_id,scene_token=data.scene_ref.scene_token,
        target_effect=data.target_effect,evidence=summary['evidence'],attack_input=None)
    options=response.get('edit_options') or []
    ego_plan = None
    if result['suitability'] == 'ready' and (options or result['edit_plan'] is not None):
        ego_plan = EgoHistoryPlan.model_validate(response.get('ego_history_plan') or {'mode': 'keep'})
        result['ego_history_plan'] = ego_plan.model_dump(mode='json', exclude_none=True)
    if result['suitability']=='ready' and options:
        for option in options:
            if option['operation']=='edit_existing' and option.get('target_track') is not None and option['target_track'] not in eligible:
                raise ValueError('Existing option target must be a fully observed valid vehicle')
        has_existing=any(option['operation']=='edit_existing' for option in options)
        has_asset=summary['evidence']['added_vehicle_asset_available']
        result['evidence']=dict(result['evidence'],option_resources=[dict(option_id=option['option_id'],
            available=option['operation']=='edit_existing' or has_asset) for option in options],
            motion_feasibility=dict(status='deferred_to_tool05'))
        if not has_existing and not has_asset:
            result['suitability']='needs_information'
            result['reasons']=[*result['reasons'],'Provide a real added-vehicle asset and size for these alternatives.']
        else:
            arguments=dict(scene_ref=data.retrieval_result_ref or data.scene_ref.model_dump(mode='json'),
                operation='auto',plan_options=options,vadv2_self_test=True,
                instruction='Construct this scene mechanism: '+data.scene_requirement+
                    '\nCorrect ego behavior to preserve for expert editing: '+data.target_effect+
                    '\nChoose or adapt a coarse alternative, test VADv2 and use its actual avoidance feedback to refine the attack. Alternatives are not hard contracts; needing to brake alone is not attack success.')
            if has_asset:arguments['vehicle']=data.vehicle.model_dump(mode='json')
            arguments.update(ego_plan.attack_arguments())
            AttackInput.model_validate(arguments)
            result['attack_input']=arguments
        return result
    if result['suitability']=='ready' and result['edit_plan'] is not None:
        plan=result['edit_plan']
        if plan['operation']=='edit_existing' and plan['target_track'] not in eligible:
            raise ValueError('Existing target must be a valid vehicle observed in every source frame, as required by tool05')
        result['evidence']=dict(result['evidence'],motion_feasibility=dict(status='deferred_to_tool05',
            scope='Original future motion does not veto coarse planning; tool05 checks generated candidates and supported takeover state.'))
        if plan['operation']=='add_vehicle' and not summary['evidence']['added_vehicle_asset_available']:
            result['suitability']='needs_information'
            result['reasons']=[*result['reasons'],'Provide a real added-vehicle PLY and length/width/height before tool05.']
        else:
            arguments=dict(scene_ref=data.retrieval_result_ref or data.scene_ref.model_dump(mode='json'),operation=plan['operation'],instruction=plan['instruction'])
            if plan['operation']=='edit_existing':arguments['target_track']=plan['target_track']
            else:arguments.update(maneuver=plan['maneuver'],vehicle=data.vehicle.model_dump(mode='json'))
            arguments.update(ego_plan.attack_arguments())
            AttackInput.model_validate(arguments)
            result['attack_input']=arguments
    return result


def validate(data,result):
    summary,eligible,_=inspect_scene(data)
    if result.edit_plan is not None and result.edit_plan.operation=='edit_existing' and result.edit_plan.target_track not in eligible:
        raise ValueError('Existing target must be a valid vehicle observed in every source frame, as required by tool05')
    for option in result.edit_options:
        if option.operation=='edit_existing' and option.target_track is not None and option.target_track not in eligible:
            raise ValueError('Existing option target must be a fully observed valid vehicle')

TOOL=LLMTool(name='edit_planner',directory=Path(__file__).parent,input_model=Input,response_model=Decision,
    output_model=Output,payload_builder=payload,enrich_result=enrich,validate_result=validate,
    client_adapter=lambda client,data:ViewClient(client,ACTIVE_OBSERVATION.get()))


def run(input_data):
    try:data=Input.model_validate({k:v for k,v in input_data.items() if k!='_llm_client'})
    except (AttributeError,ValueError,TypeError,KeyError) as error:
        return dict(status='error',tool='edit_planner',error=dict(type='input_validation_error',message='Tool input does not match its schema',validation_errors=validation_details(error)))
    try:
        cached=_cached_asset_supplement(data)
        if cached is not None:
            return cached
        inspection=inspect_scene(data)
    except (OSError,ValueError,TypeError,KeyError) as error:
        return dict(status='error',tool='edit_planner',error=dict(type='scene_inspection_error',message='Cannot inspect the retrieved scene',validation_errors=validation_details(error)))
    summary=inspection[0]
    missing=summary['evidence']['missing_resources']
    if missing:
        result=Output.model_validate(dict(scene_id=data.scene_ref.scene_id,scene_token=data.scene_ref.scene_token,
            target_effect=data.target_effect,suitability='needs_information',reasons=missing,
            edit_plan=None,attack_input=None,evidence=summary['evidence'])).model_dump(mode='json')
        return dict(status='success',tool='edit_planner',result=result,
            meta=dict(mode='real',model='resource_preflight',attempts=0,elapsed_seconds=0.0))
    token=_INSPECTION.set((data.model_dump(mode='json'),inspection))
    observation_token=ACTIVE_OBSERVATION.set(None)
    try:
        result=TOOL.execute(input_data)
        if result.get('status')=='success':
            _save_asset_supplement(data,result)
        return result
    finally:
        ACTIVE_OBSERVATION.reset(observation_token)
        _INSPECTION.reset(token)
