"""Test recovery decisions made directly by the main Agent.

The decisions suite uses synthetic state snapshots. The integration suite
uses the real planner with mock tools and one injected visual failure.
Neither suite runs a GPU backend or sends real video content.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from agent.orchestrator import Orchestrator
from agent.state import AgentState

TASK="使用提供的视频编辑现有前车，使其向自车车道移动后保持车道位置，交付验证通过的新场景。不需要策略仿真或训练。"
EDIT_ARGS={
    "video_path":"/synthetic/source.mp4","edit_config_path":"/synthetic/edit.json",
    "camera_params_path":"/synthetic/camera_params.json","coordinate_system":"smart_edit",
    "meters_to_dggt":0.01,"num_frames":40,
}
EDIT_RESULT={"edited_scene":{"rendered_video":"/synthetic/edited.mp4","scene_config":"/synthetic/edit.json"}}


def tool_result(state,name,result=None,error=None):
    envelope={"tool":name,"status":"error" if error else "success"}
    envelope["error" if error else "result"]=error if error else result
    state.add_tool_result(name,envelope)


def prepared_state():
    state=AgentState(TASK)
    tool_result(state,"reconstruction",{
        "input_dir":"/synthetic/input","input_scene_name":"000",
        "output_dir":"/synthetic/reconstruction","camera_params":"/synthetic/camera_params.json",
        "rendered_video":"/synthetic/reconstructed.mp4",
    })
    tool_result(state,"instance_discovery",{
        "instances":[{"instance_id":0,"category":"car","frames":[]}],
        "instance_info_path":"/synthetic/instance_info.json",
    })
    tool_result(state,"edit_plan_builder",{
        "edit_plan":"Move existing vehicle 0 toward the ego lane, accounting for original motion.",
        "target_instances":[0],
        "operations":[{"operation":"modify_trajectory","target":0,"parameters":{}}],
        "constraints":["Maintain road contact."],
    })
    tool_result(state,"edit_config_generator",{"saved_path":"/synthetic/edit.json"})
    return state


def scenarios():
    cases=[]
    for passed,label in ((False,"visual_failure"),(True,"validated_scene")):
        state=prepared_state()
        tool_result(state,"scene_editor",EDIT_RESULT)
        tool_result(state,"scene_validator",{
            "passed":passed,"geometry_valid":passed,"visual_valid":passed,
            "issues":[] if passed else ["Target vehicle visibly floats above the road in middle and late frames."],
            "rendered_video":"/synthetic/edited.mp4",
        })
        expected_action="finish" if passed else "call_tool"
        allowed=set() if passed else {"edit_plan_builder","edit_config_generator","scene_editor"}
        example=({"action":"finish","reason":"The requested new scene passed validation."} if passed else {
            "action":"call_tool","tool_name":"edit_plan_builder",
            "arguments":{"edit_request":"Correct road contact for vehicle 0 before rendering again.","instances":[{"instance_id":0}]},
            "reason":"Validation reports a floating vehicle; revise the edit before rendering and revalidation.",
        })
        cases.append((label,state,expected_action,allowed,example))
    for count,label in ((1,"transient_error"),(2,"repeated_transient_error")):
        state=prepared_state()
        for step in range(count):
            state.add_decision(step,{
                "action":"call_tool","tool_name":"scene_editor","arguments":EDIT_ARGS,
                "reason":"Render the requested edit.",
            })
            tool_result(state,"scene_editor",error={
                "type":"timeout","message":"DGGT worker temporarily timed out; no alternate worker or environment is available."
            })
        example=({
            "action":"call_tool","tool_name":"scene_editor","arguments":EDIT_ARGS,
            "reason":"Retry the transient timeout once, reusing the known inputs.",
        } if count==1 else {
            "action":"stop","reason":"The same timeout repeated after one retry and no alternate worker is available; the edited scene remains incomplete."
        })
        cases.append((label,state,"call_tool" if count==1 else "stop",
                      {"scene_editor"} if count==1 else set(),example))
    state=AgentState(TASK)
    tool_result(state,"reconstruction",error={
        "type":"missing_dependency",
        "message":"Required DGGT checkpoint /synthetic/model.pth does not exist. No alternate checkpoint, download, install or file-management tool is available."
    })
    cases.append(("missing_dependency",state,"stop",set(),{
        "action":"stop","reason":"Required checkpoint is missing and no available tool can supply it; reconstruction and the generated scene remain incomplete."
    }))
    return cases


class ScriptedClient:
    model="scripted-offline"
    def __init__(self,response):
        self.response=response
    def chat(self,messages):
        return json.dumps(self.response)


def run_decisions(mode):
    if mode=="real":
        from agent.llm_client import LLMClient
        client=LLMClient()
    checks=[]
    for label,state,action,allowed,example in scenarios():
        planner=Orchestrator(llm_client=client if mode=="real" else ScriptedClient(example))
        decision=planner.decide(state.task,state.to_orchestrator_dict())
        passed=decision.get("action")==action
        if allowed:
            passed=passed and decision.get("tool_name") in allowed
        if label=="transient_error":
            passed=passed and decision.get("arguments")==EDIT_ARGS
        checks.append({"case":label,"passed":passed,"decision":decision})
        print(label,"PASS" if passed else "FAIL",decision,flush=True)
    return {"suite":"decisions","mode":mode,"model":planner.llm_client.model,
            "synthetic_inputs":True,"passed":all(c["passed"] for c in checks),"checks":checks}


def run_integration(output_directory):
    from agent.config import AgentConfig
    from agent.main import Agent
    config=AgentConfig.load()
    config=replace(config,max_steps=20,
                   tool_modes={name:"mock" for name in config.tool_modes},
                   logs_root=output_directory/"agent_logs")
    agent=Agent(config)
    original=agent.executor.execute
    validation_count=0
    render_count=0
    def execute(name,arguments):
        nonlocal validation_count,render_count
        result=original(name,arguments)
        if result.get("status")!="success":
            return result
        # Contract-aligned synthetic outputs: legacy mocks omit saved_path
        # and rendered_video, which otherwise makes the planner troubleshoot
        # the fixture instead of the deliberately injected visual failure.
        if name=="edit_config_generator":
            result={**result,"result":{
                "saved_path":arguments["output_path"],
                "tool_calls":[{"name":"synthetic_config_generation","status":"success"}],
            }}
        elif name=="scene_editor":
            render_count+=1
            directory=f"/synthetic/edited/run_{render_count:02d}"
            result={**result,"result":{"edited_scene":{
                "rendered_video":directory+"/rendered.mp4",
                "scene_config":arguments.get("edit_config_path"),
                "scene_ply":directory+"/scene.ply",
                "output_dir":directory,
            }}}
        elif name=="scene_validator":
            validation_count+=1
            result=deepcopy(result)
            result["result"]["rendered_video"]=arguments.get("rendered_video")
            if validation_count==1:
                result["result"].update(
                    passed=False,geometry_valid=False,visual_valid=False,
                    issues=["Target vehicle floats above the road; revise the edit plan/configuration and render a new scene before validation."])
        return result
    agent.executor.execute=execute
    task=TASK+" 输入视频为 /synthetic/source.mp4。先重建并发现实例，不需要失败分析或检索。"
    state=agent.run(task)
    results=[e for e in state["history"] if "tool" in e]
    validations=[(i,e["result"].get("result",{}).get("passed"))
                 for i,e in enumerate(results) if e["tool"]=="scene_validator"]
    repaired=False
    if len(validations)>=2 and validations[0][1] is False and validations[-1][1] is True:
        first,last=validations[0][0],validations[-1][0]
        repaired=any(e["tool"]=="scene_editor" for e in results[first+1:last])
    validation_videos=[e["result"].get("result",{}).get("rendered_video")
                       for e in results if e["tool"]=="scene_validator"]
    new_render_validated=(len(validation_videos)>=2
                          and validation_videos[0]!=validation_videos[-1]
                          and validation_videos[-1]==f"/synthetic/edited/run_{render_count:02d}/rendered.mp4")
    errors=[{"tool":e["tool"],"error":e["result"].get("error")} for e in results if e["result"].get("status")!="success"]
    return {
        "suite":"integration","mode":"real-planner/mock-tools",
        "model":agent.orchestrator.llm_client.model,"synthetic_inputs":True,
        "passed":state["status"]=="finished" and repaired and new_render_validated and not errors,
        "status":state["status"],"tools":[e["tool"] for e in results],
        "validation_passes":[value for _,value in validations],
        "rendered_again_after_failure":repaired,
        "new_render_validated":new_render_validated,
        "validation_videos":validation_videos,"errors":errors,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["mock","real"],default="mock")
    parser.add_argument("--suite",choices=["decisions","integration"],default="decisions")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.suite=="integration" and args.mode!="real":
        parser.error("integration uses a real planner; specify --mode real")
    output=args.output or ROOT/"logs"/"agent_recovery_smoke"/(
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")+"_"+args.suite+"_"+args.mode+".json")
    output.parent.mkdir(parents=True,exist_ok=True)
    report=run_decisions(args.mode) if args.suite=="decisions" else run_integration(output.parent)
    report["timestamp"]=datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("Report:",output.resolve(),"passed:",report["passed"],flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__":
    main()
