"""Quick source-scene review with a direct, validated handoff to tool05."""
from contextvars import ContextVar
import json
import math
from pathlib import Path
from time import perf_counter

# A vehicle is only worth editing if it is close to the ego path and ahead of it once.
ELIGIBLE_MAX_DISTANCE_M = 30.0
ELIGIBLE_BEHIND_TOLERANCE_M = 10.0
from tools.llm_runtime import LLMTool, validation_details
from .schemas import Input, Decision, Output
from .observations import settings, sample_indices, Observation, ViewClient, ACTIVE_OBSERVATION
from . import observations

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
    # Being visible in every frame is not the same as being edit-relevant: a track
    # parked on another road can be fully observed and still useless. Keep only
    # vehicles that come near the ego path and are ahead of it at least once.
    relevant=set()
    for track in eligible:
        points=positions.get(track) or []
        distances=[math.hypot(x,y) for x,y in points]
        catalog[track]['nearest_ego_distance_m']=round(min(distances),1) if distances else None
        if any(d<=ELIGIBLE_MAX_DISTANCE_M and x>-ELIGIBLE_BEHIND_TOLERANCE_M for (x,y),d in zip(points,distances)):
            relevant.add(track)
        else:
            catalog[track]['edit_relevance']='too_far_or_behind'
    for track in relevant: catalog[track]['edit_relevance']='relevant'
    eligible=relevant
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
    from .mapfacts import collect as collect_map_facts
    precheck=collect_map_facts(frames,data.scenario.attack.model_dump(mode='json')) if frames else None
    # 地图缺失降级为提示，不再拦截 tool04；由 tool05 执行时检查
    candidate=data.candidate
    retrieval=dict(video_path=data.retrieval_result.video_path,source_file=data.retrieval_result.source_file,
        source_segment=data.retrieval_result.source_segment,candidate_id=data.retrieval_result.candidate_id)
    evidence=dict(annotation_path=str(frames_path),gaussian_assets=assets,frame_indices=indices,source_images=[str(p) for p in images],
        keyframes=image_evidence,visual_settings=visual_settings,eligible_vehicle_tracks=sorted(eligible),road_geometry_available=road.is_file(),
        missing_resources=missing,retrieval=retrieval,
        map_facts=(precheck or {}).get('facts'),
        map_checks=(precheck or {}).get('checks'),
        map_reasons=(precheck or {}).get('reasons'),
        map_suggestion=(precheck or {}).get('retrieval_suggestion'),
        map_evidence_limits=(precheck or {}).get('evidence_limits'),
        map_available=bool((precheck or {}).get('available')),
        target_region_frames=[row["frame_index"] for row in
                              sorted(((precheck or {}).get('facts') or {}).get('frame_target_distance') or [],
                                     key=lambda row: row.get('distance_to_target_m')
                                     if row.get('distance_to_target_m') is not None else float('inf'))[:3]
                              if row.get('distance_to_target_m') is not None],
        review_scope='Whole bound scene; source_segment is only the retrieval match, not a synchronized crop')
    summary=dict(scene_id=ref.scene_id,scene_token=ref.scene_token,matched_camera=ref.matched_camera,frame_count=len(frames),
        actors=list(catalog.values()),snapshots=snapshots,evidence=evidence,
        box_coordinates='Original source annotation coordinates; transforms are supplied where available.',
        target_scene=data.scenario.target_scene,
        attack=data.scenario.attack.model_dump(mode='json'),
        candidate_id=candidate.candidate_id,
        candidate=dict(candidate_id=candidate.candidate_id,
            base_video_requirement=candidate.base_video_requirement,
            attack_edit_strategy=candidate.attack_edit_strategy,
            ego_input_edit_strategy=candidate.ego_input_edit_strategy,
            ego_correct_trajectory_strategy=candidate.ego_correct_trajectory_strategy))
    return summary,eligible,images


def payload(data):
    summary,_,_=inspect_scene(data)
    observation=Observation(data,summary)
    ACTIVE_OBSERVATION.set(observation)
    return observation.initial_content()


def bound_result(data,summary,response):
    """Copy the scenario identity and target scene into the result; never let the model rewrite them."""
    return dict(response,
        scene_id=data.scene_ref.scene_id,scene_token=data.scene_ref.scene_token,
        candidate_id=data.candidate_id,
        target_scene=data.scenario.target_scene,
        attack=data.scenario.attack.model_dump(mode="json"),
        evidence=summary["evidence"])


def insufficient(data,summary,reasons,elapsed_seconds):
    """Missing scene prerequisites: report what is missing instead of planning on an empty scene."""
    result=bound_result(data,summary,dict(
        suitability="needs_information",reasons=list(reasons),
        checks=dict(road="unknown",lanes="unknown",space="unknown"),
        adapted_plan=None,retrieval_suggestion=None))
    return dict(status="success",tool="edit_planner",result=Output.model_validate(result).model_dump(mode="json"),
        meta=dict(mode="real",model=None,attempts=0,elapsed_seconds=elapsed_seconds))


def unsuitable_from_precheck(data,summary,checks,elapsed_seconds):
    """Trusted data verdict rejects the structure before any observation or model call."""
    reasons=list(summary["evidence"].get("map_reasons") or [])
    reasons.append("Measured structure contradicts the target scene, so no visual review or plan is produced.")
    suggestion=summary["evidence"].get("map_suggestion") or (
        "Retrieve a scene whose measured road structure supports the target scene.")
    result=bound_result(data,summary,dict(
        suitability="unsuitable",reasons=[reason for reason in reasons if reason],checks=checks,
        adapted_plan=None,retrieval_suggestion=suggestion))
    result["evidence"]=dict(result["evidence"],verification_source="measured_map_precheck")
    return dict(status="success",tool="edit_planner",
        result=Output.model_validate(result).model_dump(mode="json"),
        meta=dict(mode="real",model=None,attempts=0,elapsed_seconds=elapsed_seconds))


def enrich(data,response):
    summary,_,_=inspect_scene(data)
    result=bound_result(data,summary,response)
    precheck=summary["evidence"].get("map_checks") or {}
    conflicts={name:{"data":value,"review":result["checks"].get(name)}
               for name,value in precheck.items()
               if value in ("pass","fail") and result["checks"].get(name)!=value}
    visual=summary["evidence"].get("visual_observation") or {}
    result["evidence"]=dict(result["evidence"],
        verification_source="visual_review" if result["checks"] else "measured_map_precheck",
        visual_review=dict(cameras=sorted({sheet.get("camera") for sheet in visual.get("sheets", [])}),
                           sheet_count=len(visual.get("sheets", [])),
                           model_calls=visual.get("model_calls"),
                           checks=result["checks"], reasons=result["reasons"]))
    if conflicts:
        # A reliable measured verdict must not be silently overridden by the image review.
        result["evidence"]=dict(result["evidence"],precheck_conflicts=conflicts)
        detail="; ".join(f"{name}: data={value['data']}, review={value['review']}"
                         for name,value in conflicts.items())
        result["reasons"]=[*result["reasons"],
                           "Visual review disagrees with the measured pre-check (" + detail + "); "
                           "the scene needs re-checking before it can be planned."]
        result["adapted_plan"]=None
        if "fail" in result["checks"].values():
            result["suitability"]="unsuitable"
            result["retrieval_suggestion"]=result.get("retrieval_suggestion") or (
                "Retrieve a scene that matches the measured structure.")
        else:
            result["suitability"]="needs_information"
    return result


TOOL=LLMTool(name='edit_planner',directory=Path(__file__).parent,input_model=Input,response_model=Decision,
    output_model=Output,payload_builder=payload,enrich_result=enrich,
    client_adapter=lambda client,data:ViewClient(client,ACTIVE_OBSERVATION.get()))


def run(input_data):
    started=perf_counter()
    try:data=Input.model_validate({k:v for k,v in input_data.items() if k!='_llm_client'})
    except (AttributeError,ValueError,TypeError,KeyError) as error:
        return dict(status='error',tool='edit_planner',error=dict(type='input_validation_error',message='Tool input does not match its schema',validation_errors=validation_details(error)))
    try:
        inspection=inspect_scene(data)
    except (OSError,ValueError,TypeError,KeyError) as error:
        return dict(status='error',tool='edit_planner',error=dict(type='scene_inspection_error',message='Cannot inspect the retrieved scene',validation_errors=validation_details(error)))
    summary=inspection[0]
    if summary['evidence']['missing_resources']:
        elapsed_seconds=round(perf_counter()-started,4)
        return insufficient(data,summary,summary['evidence']['missing_resources'],elapsed_seconds)
    precheck_checks=summary['evidence'].get('map_checks') or {}
    if 'fail' in precheck_checks.values():
        elapsed_seconds=round(perf_counter()-started,4)
        return unsuitable_from_precheck(data,summary,precheck_checks,elapsed_seconds)
    token=_INSPECTION.set((data.model_dump(mode='json'),inspection))
    observation_token=ACTIVE_OBSERVATION.set(None)
    try:
        return TOOL.execute(input_data)
    finally:
        ACTIVE_OBSERVATION.reset(observation_token)
        _INSPECTION.reset(token)
