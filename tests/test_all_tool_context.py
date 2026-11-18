"""All-tool summaries preserve decisions and artifact-backed handoffs."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from agent.context import FIELDS
from agent.state import AgentState

def save(state,root,tool,payload,arguments=None,status="success"):
    folder=root/str(len(state.history))
    folder.mkdir()
    (folder/"input.json").write_text(json.dumps(arguments or {}))
    envelope=dict(status=status,tool=tool,**({"result":payload} if status=="success" else {"error":payload}))
    path=folder/"output.json";path.write_text(json.dumps(envelope))
    if arguments is not None:
        state.add_decision(len(state.history),dict(action="call_tool",tool_name=tool,arguments=arguments,reason="advance"))
    state.add_tool_result(tool,envelope,str(path))
    return str(path)

BULK=["raw numerical evidence "+str(i) for i in range(2000)]
PAYLOADS={
 "failure_analyzer":{"token":"4abb45d9ee1a5c5c","simulation":"experiment/batch","map_location":"map",
     "evidence_dir":"/logs/tool01/run-1","violation_step":8,
     "failure":{"outcome":"collision","interaction":"crossing","ego_cause":"late braking into the curve"},
     "attack":{"map_context":"t_junction","ego_action":"turn_right","ego_lane_keeping":"keep",
               "other_origin":"right_arm","other_action":"turn_left","other_lane_keeping":"keep"},
     "environment":{"time":"2021-09-16T20:16:41+00:00","weather":"cloudy","extra":BULK[0]},
     "verification":{"sample_frames":[6,8,10],"corrections":[],"notes":[]}},
 "scene_instance_generator":{"target_scene":"Curve; ego leaves road.","attack":{"map_context":"curve",
     "ego_action":"straight","ego_lane_keeping":"keep","other_origin":"none",
     "other_action":"none","other_lane_keeping":"none"},
     "candidates":[{"candidate_id":"cand_1","base_video_requirement":"left curve","retrieval_query":"day left curve clear","attack_edit_strategy":None,
                    "ego_input_edit_strategy":None,"ego_correct_trajectory_strategy":"slow before curve"},
                   {"candidate_id":"cand_2","base_video_requirement":"left curve","retrieval_query":"day left curve clear","attack_edit_strategy":None,
                    "ego_input_edit_strategy":"later arrival","ego_correct_trajectory_strategy":"slow before curve"},
                   {"candidate_id":"cand_3","base_video_requirement":"wide left curve","retrieval_query":"day wide left curve clear","attack_edit_strategy":None,
                    "ego_input_edit_strategy":None,"ego_correct_trajectory_strategy":"slow before curve"}]},
 "video_retriever":{"scene_ref":{"scene_id":"source","scene_token":"A","cameras":{"CAM_F0":{"images":BULK}}},
     "match_score":.8,"video_path":"/front.mp4"},
 "edit_planner":{"suitability":"ready","reasons":["base suitable"],"scene_id":"source","scene_token":"A",
     "candidate_id":"cand_1","target_scene":"Curve; ego leaves road.",
     "attack":{"map_context":"curve","ego_action":"straight","ego_lane_keeping":"keep",
               "other_origin":"none","other_action":"none","other_lane_keeping":"none"},
     "checks":{"road":"pass","lanes":"pass","space":"pass"},
     "adapted_plan":{"attack_edit_strategy":None,"ego_input_edit_strategy":None,
                     "ego_correct_trajectory_strategy":"slow before curve"},
     "retrieval_suggestion":None,"evidence":{"keyframes":BULK}},
 "attack_vehicle_editor":{"status":"completed","scene_id":"source","scene_token":"A","target_track":"car",
     "selected_candidate_id":"attack-1","source_scene_ref":{"frames":BULK},
     "checks":{"reference_screen":"passed","checks":{"road":"passed","render":"not_checked"},
               "physical_avoidance":"undetermined","report_path":"/checks.json"},
     "simulation_input":{"scenario_data_folder_path":"/attack","asset_folder_path":"/assets"}},
 "ego_trajectory_editor":{"status":"completed","scene_id":"source","scene_token":"A","selected_candidate_id":"ego-1",
     "expert_trajectory_path":"/expert.csv","rendered_videos":{"front":"/edited.mp4"},
     "ilft_training_data":None,"annotation_changes":{"rows":BULK}},
 "policy_runner":{"result_dir":"/simulation","rollout_dir":"/rollout","reward_dir":"/rewards",
     "candidate_rewards_present":True,"pdm_challenge":{"official_native_score":False,"pdm_defense_succeeded":False},
     "reports":[{"status":"completed","metrics":{"pdm_score":0.},"trajectories":BULK}]},
 "failure_case_filter":{"is_failure_case":True,"scene_count":100,"pdm_scores_file":"/scores.csv",
     "failure_cases":[{"scene_id":str(i),"pdm_score":0.} for i in range(100)],"scene_results":[]},
 "policy_trainer":{"checkpoint":"/updated.pth","training_mode":"rlft","log_path":"/train.log",
     "training_audit":{"finite_losses":True,"changed_parameter_count":100,"losses":{"loss":1.2},"parameter_diff":BULK}},
}

@pytest.mark.parametrize("tool",list(FIELDS))
def test_every_tool_has_an_explicit_view_without_mutating_artifacts(tmp_path,tool):
    state=AgentState("complete task")
    payload=deepcopy(PAYLOADS[tool])
    reference=save(state,tmp_path,tool,payload,arguments={"test_input":"original"})
    original=deepcopy(state.to_dict())
    context=state.to_orchestrator_dict()
    result=context["history"][-1]
    assert result["result_ref"]==reference
    assert result["context_view"]=="summary"
    assert state.to_dict()==original
    assert json.loads(Path(reference).read_text())["result"]==payload
    assert context["history"][-2]["decision"]["arguments"]=={"test_input":"original"}
    assert "arguments_ref" not in context["history"][-2]  # Small scalar inputs need no extra pointer.
    view=result["result"]["result"]
    if tool=="failure_analyzer":
        assert view["failure"]["ego_cause"]=="late braking into the curve"
    if tool=="scene_instance_generator":
        assert view==payload
    if tool=="edit_planner":
        assert view["checks"]=={"road":"pass","lanes":"pass","space":"pass"}
        assert view["adapted_plan"]["ego_correct_trajectory_strategy"]=="slow before curve"
    if tool=="attack_vehicle_editor":
        assert view["checks"]["checks"]["render"]=="not_checked"
        assert view["checks"]["physical_avoidance"]=="undetermined"
    if tool=="ego_trajectory_editor":
        assert view["rendered_videos"]==payload["rendered_videos"] and view["ilft_training_data"] is None
    if tool=="policy_runner":
        assert view["pdm_challenge"]["official_native_score"] is False
        assert view["report_summaries"][0]["metrics"]["pdm_score"]==0.
    if tool=="failure_case_filter":
        assert view["failure_cases_count"]==100 and view["failure_cases_omitted"]==92
        assert len(view["failure_cases"])==8
    if tool=="policy_trainer":
        assert view["training_audit"]["changed_parameter_count"]==100

def test_no_reference_does_not_discard_inline_input_data():
    state=AgentState("test")
    payload=PAYLOADS["video_retriever"]
    state.add_tool_result("video_retriever",dict(status="success",tool="video_retriever",result=payload))
    assert state.to_orchestrator_dict()==state.to_dict()

def test_business_refusal_and_failed_call_remain_actionable(tmp_path):
    state=AgentState("test")
    refusal=dict(status="defense_not_found",reason="No safe candidate",stop_reason="budget_exhausted",
                 failed_requirements=["braking"],attempts=[{"raw":BULK}]*10,evidence_paths=["/measurements.json"],
                 global_infeasibility_proven=False,detail={"guidance":"reduce attack speed","measurements":{"min_gap_m":-.1}})
    save(state,tmp_path,"ego_trajectory_editor",refusal)
    args={"attack_result":"/attack.json","target_effect":"brake","use_foci":False}
    save(state,tmp_path,"ego_trajectory_editor",{"type":"render_error","message":"renderer failed","log_path":"/render.log"},
         arguments=args,status="error")
    context=state.to_orchestrator_dict()
    business=context["history"][0]["result"]["result"]
    assert business["status"]=="defense_not_found"
    assert business["attempt_count"]==10
    assert business["global_infeasibility_proven"] is False
    assert business["recovery_feedback"]["guidance"]=="reduce attack speed"
    assert context["history"][-2]["decision"]["arguments"]==args
    assert context["history"][-1]["result"]["error"]["log_path"]=="/render.log"

def test_mixed_pipeline_context_is_smaller_and_keeps_goal_and_checkpoint(tmp_path):
    state=AgentState("generation and training")
    for tool,payload in PAYLOADS.items():
        save(state,tmp_path,tool,deepcopy(payload),arguments={"evidence":BULK})
    full=json.dumps(state.to_dict(),ensure_ascii=False)
    compact=json.dumps(state.to_orchestrator_dict(),ensure_ascii=False)
    assert len(compact)<len(full)*.2
    assert "left curve" in compact and "slow before curve" in compact and "/updated.pth" in compact
    assert "raw numerical evidence 1999" not in compact

def test_reference_handoffs_are_accepted_by_existing_tool_schemas(tmp_path):
    from tools.tool02_scene_instance_generator.schemas import Input as RequirementInput
    from tools.tool04_edit_planner.schemas import Input as PlannerInput
    from mcp_servers.scene_editing.adapter.schemas import AttackInput,EgoInput
    from mcp_servers.scene_editing.adapter.mock import attack
    state=AgentState("test")
    save(state,tmp_path,"failure_analyzer",PAYLOADS["failure_analyzer"])
    analysis=state.to_orchestrator_dict()["history"][-1]["result"]["result"]
    RequirementInput.model_validate({"failure_analysis":analysis})
    scene=dict(scene_id="source",scene_token="A",dataset="test",sample_root=str(tmp_path),
        matched_camera="CAM_F0",frame_count=12,fps=2.,manifest_path="/manifest.json",
        frames_path="/frames.json",cameras_path="/cameras.json",cameras={},images_complete=False)
    retrieval={"scene_ref":scene,"video_path":"/front.mp4","mode":"real","match_score":.8,
        "candidate_id":"cand_1"}
    retrieval_ref=save(state,tmp_path,"video_retriever",retrieval)
    from tools.tool02_scene_instance_generator.schemas import Output as ScenarioOutput
    scenario=ScenarioOutput.model_validate(dict(
        target_scene="Curve; ego leaves road.",
        attack={"map_context":"curve","ego_action":"straight","ego_lane_keeping":"keep",
                "other_origin":"none","other_action":"none","other_lane_keeping":"none"},
        candidates=[dict(candidate_id=f"cand_{i}",base_video_requirement="left curve",
                         retrieval_query="day left curve clear",
                         attack_edit_strategy=None,ego_input_edit_strategy=None,
                         ego_correct_trajectory_strategy="slow before curve") for i in (1,2,3)])).model_dump(mode="json")
    save(state,tmp_path,"edit_planner",dict(suitability="ready",reasons=["base suitable"],
        scene_id="A",scene_token="A",candidate_id="cand_1",target_scene=scenario["target_scene"],
        attack=scenario["attack"],checks={"road":"pass","lanes":"pass","space":"pass"},
        adapted_plan={"attack_edit_strategy":None,"ego_input_edit_strategy":None,
                      "ego_correct_trajectory_strategy":"slow before curve"},
        retrieval_suggestion=None,evidence={"keyframes":[]}))
    compact=state.to_orchestrator_dict()
    view=compact["history"][-1]["result"]["result"]
    assert view["adapted_plan"]["ego_correct_trajectory_strategy"]=="slow before curve"
    PlannerInput.model_validate(dict(scenario_result=scenario,candidate_id="cand_1",
        retrieval_result=retrieval_ref)).scene_ref.scene_token=="A"
    plan_args=dict(scene_ref=scene,operation="edit_existing",target_track="car",instruction="slow lead car")
    attack_result=attack(plan_args)
    attack_ref=save(state,tmp_path,"attack_vehicle_editor",attack_result)
    state.add_decision(6,dict(tool_name="ego_trajectory_editor",arguments={"attack_result":attack_result,"target_effect":"brake"}))
    args=state.to_orchestrator_dict()["history"][-1]["decision"]["arguments"]
    assert args["attack_result"]==attack_ref
    assert EgoInput.model_validate(args).attack_result.scene_token=="A"


def test_successful_simulation_inputs_keep_training_base_checkpoint(tmp_path):
    state=AgentState("train after evaluation")
    args={"checkpoint_path":"/base.pth","config_path":"/policy.py",
          "scenario_data_folder_path":"/scenarios","asset_folder_path":"/assets"}
    save(state,tmp_path,"policy_runner",PAYLOADS["policy_runner"],arguments=args)
    assert state.to_orchestrator_dict()["history"][0]["decision"]["arguments"]==args

def test_mock_flags_and_error_tail_are_not_lost(tmp_path):
    state=AgentState("test")
    save(state,tmp_path,"policy_trainer",dict(PAYLOADS["policy_trainer"],mock=True))
    save(state,tmp_path,"policy_runner",{"type":"execution_error","message":"trace start "+("a"*3000)+" FileNotFound: /missing.map"},status="error")
    compact=state.to_orchestrator_dict()
    assert compact["history"][0]["result"]["result"]["mock"] is True
    error=compact["history"][-1]["result"]["error"]
    assert "/missing.map" in error["message"]
    assert error["message_omitted_characters"]>0
