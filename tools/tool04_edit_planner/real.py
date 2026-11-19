"""Quick source-scene review with a direct, validated handoff to tool05."""
import json
import math
from pathlib import Path
from time import perf_counter

# A vehicle is only worth editing if it is close to the ego path and ahead of it once.
ELIGIBLE_MAX_DISTANCE_M = 30.0
ELIGIBLE_BEHIND_TOLERANCE_M = 10.0
CANONICAL_VEHICLE_ASSET = Path(__file__).resolve().parents[2] / "data/assets/vehicle_01.ply"
CANONICAL_VEHICLE_SIZE_M = [4.432, 2.033, 1.637]
from tools.llm_runtime import validation_details
from .schemas import Input, Output

def sample_indices(count,limit):
    if count<=limit:return list(range(count))
    return [round(i*(count-1)/(limit-1)) for i in range(limit)]


def inspect_scene(data):
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
    indices=sample_indices(len(frames),min(12,len(frames))) if frames else []
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
    manifest=Path(ref.manifest_path).expanduser()
    if manifest.is_file():
        value=json.loads(manifest.read_text())
        if isinstance(value,dict) and 'timestamps_us' in value:
            if value.get('scene_token')!=ref.scene_token or value['timestamps_us']!=[f['timestamp'] for f in frames]:
                raise ValueError('Retrieval manifest belongs to a different scene or timeline')
    road=Path(ref.sample_root)/'metadata/map_drivable_official.json'
    from .mapfacts import collect as collect_map_facts
    precheck=collect_map_facts(frames,data.scenario.attack.model_dump(mode='json')) if frames else None
    # 地图缺失降级为提示，不再拦截 tool04；由 tool05 执行时检查
    candidate=data.candidate
    retrieval=dict(video_path=data.retrieval_result.video_path,source_file=data.retrieval_result.source_file,
        source_segment=data.retrieval_result.source_segment,candidate_id=data.retrieval_result.candidate_id)
    evidence=dict(annotation_path=str(frames_path),gaussian_assets=assets,frame_indices=indices,source_images=[],
        eligible_vehicle_tracks=sorted(eligible),road_geometry_available=road.is_file(),
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
        actors=list(catalog.values()),evidence=evidence,
        box_coordinates='Original source annotation coordinates; transforms are supplied where available.',
        target_scene=data.scenario.target_scene,
        attack=data.scenario.attack.model_dump(mode='json'),
        candidate_id=candidate.candidate_id,
        candidate=dict(candidate_id=candidate.candidate_id,
            base_video_requirement=candidate.base_video_requirement,
            retrieval_query=candidate.retrieval_query,
            attack_edit_strategy=candidate.attack_edit_strategy,
            ego_input_edit_strategy=candidate.ego_input_edit_strategy,
            ego_correct_trajectory_strategy=candidate.ego_correct_trajectory_strategy))
    return summary,eligible,[]


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
    result["evidence"]=dict(result["evidence"],visual_model_used=False)
    return dict(status="success",tool="edit_planner",result=Output.model_validate(result).model_dump(mode="json"),
        meta=dict(mode="real",model=None,attempts=0,elapsed_seconds=elapsed_seconds))


def unsuitable_from_precheck(data,summary,checks,elapsed_seconds):
    """Trusted data verdict rejects the structure before any observation or model call."""
    reasons=list(summary["evidence"].get("map_reasons") or [])
    reasons.append("Measured structure contradicts the target scene, so no plan is produced.")
    suggestion=summary["evidence"].get("map_suggestion") or (
        "Retrieve a scene whose measured road structure supports the target scene.")
    result=bound_result(data,summary,dict(
        suitability="unsuitable",reasons=[reason for reason in reasons if reason],checks=checks,
        adapted_plan=None,retrieval_suggestion=suggestion))
    result["evidence"]=dict(result["evidence"],verification_source="measured_map_precheck",
                            visual_model_used=False)
    return dict(status="success",tool="edit_planner",
        result=Output.model_validate(result).model_dump(mode="json"),
        meta=dict(mode="real",model=None,attempts=0,elapsed_seconds=elapsed_seconds))


def _path_plan(facts,candidate):
    pairs=((facts or {}).get('interaction_path_pairs') or [])
    chosen=next((pair for pair in pairs if pair.get('interaction_compatible') is True),None)
    if chosen is None:
        return candidate.ego_input_edit_strategy or "沿当前可通行车道调整自车接近速度和到达时机，形成目标交互。"
    ego_kind=str(chosen.get('ego_path_kind') or '')
    other_id=chosen.get('other_connector_fid')
    if ego_kind.startswith('approach_straight:'):
        lane=ego_kind.split(':',1)[1]
        route=f"沿车道 {lane} 的现有直行接近路径"
    elif ego_kind.startswith('turn:'):
        _,turn,connector=ego_kind.split(':',2)
        turn_name={'left':'左转','right':'右转','straight':'直行'}.get(turn,turn)
        route=f"使用现有 connector {connector} 在目标阶段{turn_name}进入路口"
    else:
        route="使用现有车道连接进入目标交互区域"
    detail=candidate.ego_input_edit_strategy or "调整自车接近速度与到达时刻，保持正常行驶速度。"
    return f"{route}，与右侧支路 connector {other_id} 的目标车辆形成交互；{detail}"


def _correct_plan(facts,candidate):
    pairs=((facts or {}).get('interaction_path_pairs') or [])
    chosen=next((pair for pair in pairs if pair.get('interaction_compatible') is True),None)
    kind=str((chosen or {}).get('ego_path_kind') or '')
    if kind.startswith('turn:'):
        turn=kind.split(':',2)[1]
        turn_name={'left':'左转','right':'右转'}.get(turn,'转弯')
        return f"危险出现时先减速或停车让行，确认安全后沿现有道路正常{turn_name}驶出。"
    return candidate.ego_correct_trajectory_strategy


def _tool05_input(data, summary, plan):
    """Build the only executable bridge from tool04's map review to tool05."""
    if data.scenario.attack.other_origin == "none":
        return None
    eligible = list(summary["evidence"].get("eligible_vehicle_tracks") or [])
    attack_strategy = plan.get("attack_edit_strategy") or "构造目标车辆交互。"
    options = []
    if eligible:
        options.append(dict(
            option_id="edit_existing",
            operation="edit_existing",
            instruction=("在场景中已验证可编辑的车辆中，选择最接近目标交互的一辆；"
                         + attack_strategy),
            rationale="工具④已确认该场景存在持续可见且接近自车路径的真实车辆。",
            uncertainty="具体车辆、轨迹和到达时机由工具⑤搜索验证。"))
    vehicle = data.vehicle
    vehicle_dict = vehicle.model_dump(mode="json") if vehicle is not None else None
    if vehicle_dict is None and CANONICAL_VEHICLE_ASSET.is_file():
        vehicle_dict = dict(asset_path=str(CANONICAL_VEHICLE_ASSET),
                            size_m=list(CANONICAL_VEHICLE_SIZE_M))
    if vehicle_dict is not None:
        maneuver = ("stationary" if
                    data.scenario.attack.other_origin == "ahead_same_lane"
                    and data.scenario.attack.other_action == "straight"
                    else "left_cut_in")
        options.append(dict(
            option_id="add_vehicle",
            operation="add_vehicle",
            maneuver=maneuver,
            instruction=("在当前场景可编辑的道路空间内新增目标车辆，形成目标交互；"
                         + attack_strategy),
            rationale="使用项目提供的真实车辆资产构造场景中不存在的目标车辆。",
            uncertainty="新增位置、轨迹和到达时机必须由工具⑤搜索并验证。"))
    if not options:
        return None
    operation = options[0]["operation"] if len(options) == 1 else "auto"
    request = dict(
        scene_ref=data.scene_ref.model_dump(mode="json"),
        instruction=data.scenario.target_scene + " " + attack_strategy,
        operation=operation,
        plan_options=[] if operation != "auto" else options,
        prepare_ego_history=plan.get("ego_input_edit_strategy") is not None,
        vadv2_self_test=True,
    )
    if operation == "add_vehicle":
        request.update(maneuver=options[0]["maneuver"], vehicle=vehicle_dict)
    elif operation == "auto":
        request["vehicle"] = vehicle_dict
    from mcp_servers.scene_editing.adapter.schemas import AttackInput
    return AttackInput.model_validate(request).model_dump(mode="json")


def ready_from_precheck(data,summary,checks,elapsed_seconds):
    """Data-only ready verdict and plan; no image or model call is used."""
    facts=summary['evidence'].get('map_facts') or {}
    candidate=data.candidate
    attack_plan=(None if data.scenario.attack.other_origin=='none'
                 else candidate.attack_edit_strategy or "编辑右侧支路车辆，使其在目标阶段进入自车路径。")
    plan=dict(attack_edit_strategy=attack_plan,
              ego_input_edit_strategy=_path_plan(facts,candidate),
              ego_correct_trajectory_strategy=_correct_plan(facts,candidate))
    reasons=list(summary['evidence'].get('map_reasons') or [])
    reasons.append("Map and annotation evidence pass road, lane and placement checks; no visual model was used.")
    handoff = _tool05_input(data, summary, plan)
    if data.scenario.attack.other_origin != "none" and handoff is None:
        result = bound_result(data, summary, dict(
            suitability="needs_information",
            reasons=reasons + ["No executable existing vehicle or added-vehicle asset is available for tool05."],
            checks=dict(road=checks.get("road", "unknown"), lanes=checks.get("lanes", "unknown"),
                        space=checks.get("space", "unknown")),
            adapted_plan=None, retrieval_suggestion=None, attack_input=None))
        result["evidence"] = dict(result["evidence"], verification_source="measured_map_precheck",
                                  visual_model_used=False)
        return dict(status="success", tool="edit_planner",
                    result=Output.model_validate(result).model_dump(mode="json"),
                    meta=dict(mode="real", model=None, attempts=0, elapsed_seconds=elapsed_seconds))
    result=bound_result(data,summary,dict(
        suitability="ready",reasons=reasons,checks=checks,adapted_plan=plan,
        retrieval_suggestion=None,attack_input=handoff))
    result['evidence']=dict(result['evidence'],verification_source="measured_map_precheck",
                            visual_model_used=False)
    return dict(status="success",tool="edit_planner",
        result=Output.model_validate(result).model_dump(mode="json"),
        meta=dict(mode="real",model=None,attempts=0,elapsed_seconds=elapsed_seconds))


def needs_information_from_precheck(data,summary,checks,elapsed_seconds):
    reasons=list(summary['evidence'].get('map_reasons') or [])
    reasons.extend(summary['evidence'].get('map_evidence_limits') or [])
    reasons.append("Map or annotation evidence is incomplete; no visual model fallback is configured.")
    result=bound_result(data,summary,dict(
        suitability="needs_information",reasons=reasons,checks=checks,
        adapted_plan=None,retrieval_suggestion=None))
    result['evidence']=dict(result['evidence'],verification_source="measured_map_precheck",
                            visual_model_used=False)
    return dict(status="success",tool="edit_planner",
        result=Output.model_validate(result).model_dump(mode="json"),
        meta=dict(mode="real",model=None,attempts=0,elapsed_seconds=elapsed_seconds))


def _run_scene(data,started):
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
    elapsed_seconds=round(perf_counter()-started,4)
    if precheck_checks == dict(road='pass',lanes='pass',space='pass'):
        return ready_from_precheck(data,summary,precheck_checks,elapsed_seconds)
    return needs_information_from_precheck(data,summary,precheck_checks,elapsed_seconds)


def _alternative_input(data,alternative):
    """Build the same review against one already-bound fallback scene."""
    payload=data.model_dump(mode='json')
    retrieval=dict(payload['retrieval_result'])
    retrieval.update(scene_ref=alternative.scene_ref.model_dump(mode='json'),
                     video_path=alternative.video_path,source_file=alternative.source_file,
                     source_segment=alternative.source_segment,alternatives=[])
    payload['retrieval_result']=retrieval
    return Input.model_validate(payload)


def _attempt_note(rank,outcome,data):
    if outcome.get('status')=='success':
        result=outcome['result']
        return dict(rank=rank,scene_id=result['scene_id'],scene_token=result['scene_token'],
                    suitability=result['suitability'],checks=result['checks'],reasons=result['reasons'])
    return dict(rank=rank,scene_id=data.scene_ref.scene_id,scene_token=data.scene_ref.scene_token,
                error=outcome.get('error'))


def _with_attempts(outcome,attempts,selected_rank=None):
    if outcome.get('status')=='success':
        evidence=outcome['result'].setdefault('evidence',{})
        evidence['retrieval_attempts']=attempts
        evidence['selected_rank']=selected_rank
    return outcome


def run(input_data):
    started=perf_counter()
    try:data=Input.model_validate({k:v for k,v in input_data.items() if k!='_llm_client'})
    except (AttributeError,ValueError,TypeError,KeyError) as error:
        return dict(status='error',tool='edit_planner',error=dict(type='input_validation_error',message='Tool input does not match its schema',validation_errors=validation_details(error)))
    queue=[(1,data)]
    queue.extend((item.rank,_alternative_input(data,item))
                 for item in sorted(data.retrieval_result.alternatives,key=lambda item:item.rank))
    attempts=[];first_failure=None
    for rank,current in queue:
        attempt_started=started if rank==1 else perf_counter()
        outcome=_run_scene(current,attempt_started)
        attempts.append(_attempt_note(rank,outcome,current))
        if outcome.get('status')=='success' and outcome['result']['suitability']=='ready':
            return _with_attempts(outcome,attempts,rank)
        if first_failure is None:
            first_failure=outcome
    return _with_attempts(first_failure,attempts,None)
