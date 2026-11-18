"""Deterministic, tool-specific decision views; full artifacts stay on disk."""
from copy import deepcopy
from pathlib import Path
import json

# Fields needed for orchestration, completion, recovery and the next tool.
FIELDS = {
    "failure_analyzer": ("token","violation_step","simulation","map_location","evidence_dir","failure","attack",
                         "environment","verification"),
    "scene_instance_generator": ("target_scene","attack","candidates"),
    "video_retriever": ("video_path","source_file","source_segment","match_score","mode","candidate_id"),
    "edit_planner": ("suitability","reasons","scene_id","scene_token","candidate_id","target_scene",
                     "attack","checks","adapted_plan","retrieval_suggestion"),
    "attack_vehicle_editor": ("status","stage","mode","run_dir","scene_id","scene_token","operation","target_track","selected_candidate_id","frozen_scene_path","frames_path","trajectory_path","provenance_path","submission_path","simulation_input","rendered_videos","pdm_probe_result_path","vadv2_probe_result_path","policy_collision_report"),
    "ego_trajectory_editor": ("status","stage","mode","run_dir","scene_id","scene_token","attack_run_dir","target_track","selected_candidate_id","selected_source","expert_trajectory_path","trajectory_metadata_path","conflict_brief_path","submission_path","simulation_input","expert_motion_summary","ilft_training_data","rendered_videos"),
    "policy_runner": ("result_dir","policy_result","collision_report","evaluation_scope","pdm_subscores","policy_behavior_summary","rollout_dir","reward_dir","candidate_rewards_present","rlft_training_data","rlft_data_report","pdm_challenge"),
    "failure_case_filter": ("is_failure_case","pdm_scores_file","scene_count","failure_cases","scene_results"),
    "policy_trainer": ("checkpoint","training_mode","log_path","work_dir","epochs","final_loss","training_config","export_dir"),
}
SCENE_FIELDS = ("scene_id","scene_token","dataset","sample_root","matched_camera","frame_count","fps","images_complete")
BUSINESS_FIELDS = ("reason","stop_reason","failed_requirements","failed_requirements_scope",
                   "evidence_paths","scope","global_infeasibility_proven")

def pick(value, fields):
    return {k:deepcopy(value[k]) for k in fields if k in value}

def bounded_rows(rows, limit=8):
    if not isinstance(rows,list):
        return deepcopy(rows)
    return deepcopy(rows[:limit])

def check_view(value):
    if not isinstance(value,dict):
        return deepcopy(value)
    return pick(value, ("reference_screen","checks","report_path","physical_avoidance"))

def brief_metrics(value):
    if not isinstance(value,dict):
        return deepcopy(value)
    # Scalar counts and flags preserve actual measurements; tensors/log histories do not.
    return {k:deepcopy(v) for k,v in value.items() if v is None or isinstance(v,(str,int,float,bool))}

def summarize(tool, payload):
    result = pick(payload,FIELDS[tool])
    result.update(pick(payload,("mode","mock","status")))
    if tool=="video_retriever":
        scene=payload.get("scene_ref")
        if isinstance(scene,dict):
            result["scene_ref"]=pick(scene,SCENE_FIELDS)
            cameras=scene.get("cameras")
            if isinstance(cameras,dict):
                result["available_cameras"]=list(cameras)
        else:
            result["scene_ref"]=deepcopy(scene)
    elif tool=="edit_planner":
        evidence=payload.get("evidence")
        if isinstance(evidence,dict):
            result["evidence"]=pick(evidence,("map_available","map_checks","map_reasons",
                "map_evidence_limits","target_region_frames","verification_source","precheck_conflicts"))
            for key in ("map_reasons","map_evidence_limits"):
                if isinstance(result["evidence"].get(key),list):
                    result["evidence"][key]=bounded_rows(result["evidence"][key],6)
            facts=evidence.get("map_facts")
            if isinstance(facts,dict):
                result["evidence"]["map_facts"]=pick(facts,("selection_policy","target_junction_fid",
                    "target_entry_distance_m","target_entry_frame_index","target_junction",
                    "approach_lane_count","lanes_with_direction","connectors_with_turn","connector_count"))
            visual=evidence.get("visual_review")
            if isinstance(visual,dict):
                result["evidence"]["visual_review"]=pick(visual,("cameras","sheet_count","model_calls",
                    "checks","reasons"))
            observation=evidence.get("visual_observation")
            if isinstance(observation,dict) and observation.get("record_dir"):
                result["evidence"]["observation_record_dir"]=observation["record_dir"]
    elif tool in ("attack_vehicle_editor","ego_trajectory_editor"):
        if "checks" in payload:
            result["checks"]=check_view(payload["checks"])
        result.update(pick(payload,BUSINESS_FIELDS))
        if "attempts" in payload:
            result["attempt_count"]=len(payload["attempts"])
        # Keep measured recovery feedback, without archived internal requests.
        detail=payload.get("detail")
        if isinstance(detail,dict):
            result["recovery_feedback"]=pick(detail,("status","reason","stop_reason","revision_feedback",
                "guidance","failed_requirements","measurements","reuse_completed_from",
                "termination_evidence","request_count","attempt_count","remaining_budget"))
    elif tool=="policy_runner":
        reports=payload.get("reports")
        if isinstance(reports,list):
            result["report_count"]=len(reports)
            # Only explicit scalar summaries; no candidate rewards or rollout arrays.
            result["report_summaries"]=[]
            for row in reports[:3]:
                item=brief_metrics(row)
                if isinstance(row,dict):
                    for key in ("metrics","summary","scores"):
                        if isinstance(row.get(key),dict):
                            item[key]=brief_metrics(row[key])
                result["report_summaries"].append(item)
            if len(reports)>3:
                result["reports_omitted"]=len(reports)-3
    elif tool=="policy_trainer":
        audit=payload.get("training_audit")
        if isinstance(audit,dict):
            result["training_audit"]=pick(audit,("finite_losses","changed_parameter_count"))
            if isinstance(audit.get("losses"),dict):
                result["training_audit"]["losses"]=brief_metrics(audit["losses"])
        prep=payload.get("data_preparation")
        if isinstance(prep,dict):
            result["data_preparation"]=brief_metrics(prep)
    if tool=="failure_case_filter":
        for field in ("failure_cases","scene_results"):
            if isinstance(payload.get(field),list):
                rows=payload[field]
                result[field]=bounded_rows(rows)
                result[field+"_count"]=len(rows)
                if len(rows)>8:
                    result[field+"_omitted"]=len(rows)-8
    return result

def compact_state(source):
    state=deepcopy(source)
    references=[]
    decisions=[]
    for entry in state["history"]:
        decision=entry.get("decision")
        if isinstance(decision,dict):
            decisions.append(entry)
            args=decision.get("arguments",{})
            if isinstance(args,dict):
                replace_references(decision.get("tool_name"),args,references)
        tool=entry.get("tool")
        envelope=entry.get("result")
        if not tool or not isinstance(envelope,dict):
            continue
        payload=envelope.get("result")
        reference=entry.get("result_ref")
        # Never discard data without a full saved result to hand downstream.
        if reference and isinstance(payload,dict) and envelope.get("status")=="success" and tool in FIELDS:
            original=deepcopy(payload)
            entry["result"]={**pick(envelope,("status","tool","job")), "result":summarize(tool,payload)}
            if isinstance(envelope.get("meta"),dict):
                entry["result"]["meta"]=pick(envelope["meta"],("mode","model","elapsed_seconds","attempts"))
            entry["context_view"]="summary"
            entry["result"]["result_ref"]=reference
            hint=HANDOFF_HINTS.get(tool)
            if hint:
                entry["result"]["handoff_hint"]=hint
            references.append((tool,original,reference))
        elif reference and envelope.get("status")=="error":
            error=envelope.get("error",{})
            if isinstance(error,dict):
                entry["result"]=pick(envelope,("status","tool","job"))
                entry["result"]["error"]=pick(error,("type","message","validation_errors","log_path","run_dir","stage","retryable",
                                                    "failure_stage","backend_error_type","archive_validated","scene_validated",
                                                    "error_details","recovery_hint"))
                message=entry["result"]["error"].get("message")
                if isinstance(message,str) and len(message)>1600:
                    entry["result"]["error"]["message"]=message[:800]+" [middle omitted; see result_ref] "+message[-800:]
                    entry["result"]["error"]["message_omitted_characters"]=len(message)-1600
                # Unstructured diagnostics remain available in the original result_ref.
        # Successful historical calls don't need a second copy of their inputs.
        # Retain failed-call inputs verbatim for corrective retries.
        if envelope.get("status")=="success" and reference:
            previous=next((d for d in reversed(decisions) if d["decision"].get("tool_name")==tool and not d.get("arguments_ref")),None)
            input_path=Path(reference).parent/"input.json"
            if previous is not None and input_path.is_file():
                original_args=previous["decision"].get("arguments",{})
                kept,omitted={},[]
                for key,value in original_args.items():
                    if value is None or isinstance(value,(str,int,float,bool)) or len(json.dumps(value,ensure_ascii=False))<=2048:
                        kept[key]=deepcopy(value)
                    else:
                        omitted.append(key)
                previous["decision"]["arguments"]=kept
                if kept!=original_args:
                    previous["arguments_ref"]=str(input_path)
                if omitted:
                    previous["argument_fields_omitted"]=omitted
    return state

HANDOFF_HINTS = {
    "video_retriever": "Handoff: pass result_ref verbatim as edit_planner.retrieval_result or attack_vehicle_editor.scene_ref; never retype this JSON.",
    "edit_planner": "Handoff: ready results carry adapted_plan; do not read removed edit_plan or attack_input.",
    "attack_vehicle_editor": "Handoff: pass result_ref verbatim as ego_trajectory_editor.attack_result; never retype this JSON.",
}

def replace_references(tool,args,references):
    field,expected = {
        "edit_planner":("retrieval_result","video_retriever"),
        "attack_vehicle_editor":("scene_ref","video_retriever"),
        "ego_trajectory_editor":("attack_result","attack_vehicle_editor"),
    }.get(tool,(None,None))
    if field not in args or not isinstance(args.get(field),dict):
        return
    value=args[field]
    if value.get("status")=="success" and value.get("tool")==expected:
        value=value.get("result")
    for name,payload,reference in reversed(references):
        if name!=expected:
            continue
        matches=value==payload
        if expected=="video_retriever":
            matches=matches or value==payload.get("scene_ref") or value=={"scene_ref":payload.get("scene_ref")}
        if matches:
            args[field]=reference
            return
