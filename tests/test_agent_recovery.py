"""Recovery belongs to the main orchestrator, not a separate LLM tool."""
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.config import AgentConfig
from agent.main import Agent
from agent.orchestrator import Orchestrator
from agent.schemas import validate_decision
from agent.tool_descriptions import get_tool_descriptions
from tools.registry import MOCK_TOOL_REGISTRY, REAL_TOOL_REGISTRY, call_tool


class DecisionClient:
    model="offline-planner"
    base_url="https://example.invalid/v1"

    def __init__(self,decisions):
        self.decisions=list(decisions)
        self.messages=[]

    def chat(self,messages):
        self.messages.append(deepcopy(messages))
        return json.dumps(self.decisions.pop(0))


def call(name,arguments):
    return {"action":"call_tool","tool_name":name,"arguments":arguments,"reason":"Follow the supplied evidence."}


def make_agent(tmp_path,decisions):
    client=DecisionClient(decisions)
    orchestrator=Orchestrator(llm_client=client)
    config=AgentConfig.load()
    config=replace(config,tool_modes={name:"mock" for name in config.tool_modes},logs_root=tmp_path)
    with patch("agent.main.Orchestrator",return_value=orchestrator):
        agent=Agent(config)
    return agent,client


def test_registry_descriptions_and_configuration_expose_only_13_tools():
    names={item["name"] for item in get_tool_descriptions()}
    assert len(names)==13
    assert names==set(MOCK_TOOL_REGISTRY)==set(REAL_TOOL_REGISTRY)==set(AgentConfig.load().tool_modes)
    assert "recovery_decider" not in names
    assert not Path("tools/tool11_recovery_decider").exists()
    with pytest.raises(ValueError,match="Unknown tool"):
        call_tool("recovery_decider",{},mode="mock")


def test_removed_tool_mode_is_not_silently_accepted(tmp_path):
    config=json.loads(Path("configs/agent.json").read_text())
    config["tools"]["modes"]["recovery_decider"]="real"
    path=tmp_path/"config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="unknown tools"):
        AgentConfig.load(path)


def test_main_agent_receives_failed_validation_and_runs_repair_then_revalidation(tmp_path):
    decisions=[
        call("scene_validator",{"rendered_video":"/synthetic/old.mp4","scene_requirement":"A grounded vehicle.","mock_force_fail":True}),
        call("edit_plan_builder",{"edit_request":"Correct the floating vehicle using the supplied road-contact evidence.","instances":[{"instance_id":0}]}),
        call("scene_editor",{"video_path":"/synthetic/source.mp4","edit_config_path":"/synthetic/revised.json"}),
        call("scene_validator",{"rendered_video":"/synthetic/revised.mp4","scene_requirement":"A grounded vehicle."}),
        {"action":"finish","reason":"The revised scene passed validation."},
    ]
    agent,client=make_agent(tmp_path,decisions)
    state=agent.run("Generate and validate a scene; revise failed edits.")
    results=[entry for entry in state["history"] if "tool" in entry]
    assert state["status"]=="finished"
    assert results[0]["result"]["result"]["passed"] is False
    assert results[-1]["result"]["result"]["passed"] is True
    assert [entry["tool"] for entry in results]==[
        "scene_validator","edit_plan_builder","scene_editor","scene_validator"]
    repair_context=json.loads(client.messages[1][1]["content"])
    assert repair_context["current_state"]["history"][-1]["result"]["result"]["passed"] is False
    assert "recovery_decider" not in {tool["name"] for tool in repair_context["available_tools"]}
    assert len(client.messages)==5


def test_repeated_transient_failure_stops_as_error_instead_of_success(tmp_path):
    arguments={"video_path":"/synthetic/source.mp4","edit_config_path":"/synthetic/edit.json"}
    decisions=[
        call("scene_editor",arguments),
        call("scene_editor",arguments),
        {"action":"stop","reason":"Rendering timed out again after one retry; no supported alternative is available."},
    ]
    agent,client=make_agent(tmp_path,decisions)
    error={"status":"error","tool":"scene_editor","error":{"type":"timeout","message":"Transient rendering timeout"}}
    with patch.object(agent.executor,"execute",return_value=error) as execute:
        state=agent.run("Generate a valid edited scene.")
    assert execute.call_count==2
    assert state["status"]=="error"
    assert "timed out" in state["error"]
    assert state["history"][-1]["decision"]["action"]=="stop"
    final=json.loads(agent.logger.final_state_path.read_text())
    assert final["status"]=="error"
    assert final["error"]==state["error"]
    assert json.loads(client.messages[-1][1]["content"])["current_state"]["history"][-1]["result"]["status"]=="error"


def test_explicit_pass_finishes_without_recovery_or_unrequested_simulation(tmp_path):
    agent,client=make_agent(tmp_path,[
        call("scene_validator",{"rendered_video":"/synthetic/scene.mp4","scene_requirement":"Requested scene."}),
        {"action":"finish","reason":"The requested scene passed validation."},
    ])
    state=agent.run("Generate a validated scene only; do not simulate or train.")
    assert state["status"]=="finished"
    assert [e["tool"] for e in state["history"] if "tool" in e]==["scene_validator"]


def test_removed_recovery_tool_call_is_corrected_to_a_direct_action():
    client=DecisionClient([
        call("recovery_decider",{"evaluation":{"passed":False}}),
        call("edit_plan_builder",{"edit_request":"Correct road contact."}),
    ])
    planner=Orchestrator(llm_client=client)
    result=planner.decide("Repair the failed edit.",{"history":[]})
    assert result["tool_name"]=="edit_plan_builder"
    assert len(client.messages)==2
    assert "unknown tool_name" in client.messages[-1][-1]["content"]


def test_stop_is_a_valid_decision_and_requires_a_reason():
    names={item["name"] for item in get_tool_descriptions()}
    assert validate_decision({"action":"stop","reason":"Required checkpoint is unavailable."},names)==(True,[])
    valid,errors=validate_decision({"action":"stop","reason":""},names)
    assert not valid
    assert "reason must be a non-empty string" in errors


def test_invalid_stop_is_corrected_in_the_main_orchestrator():
    client=DecisionClient([
        {"action":"stop","reason":""},
        {"action":"stop","reason":"The required checkpoint is missing and no tool can install it."},
    ])
    result=Orchestrator(llm_client=client).decide("Reconstruct the scene.",{"history":[]})
    assert result["action"]=="stop"
    assert "checkpoint" in result["reason"]
    assert "stop format" in client.messages[-1][-1]["content"]
