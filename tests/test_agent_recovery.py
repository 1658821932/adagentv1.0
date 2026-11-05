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


def test_registry_descriptions_and_configuration_expose_only_9_tools():
    names={item["name"] for item in get_tool_descriptions()}
    assert len(names)==9
    assert names==set(MOCK_TOOL_REGISTRY)==set(REAL_TOOL_REGISTRY)==set(AgentConfig.load().tool_modes)
    assert "recovery_decider" not in names
    assert "scene_instance_generator" in names
    assert "scene_requirement_generator" not in names
    assert not Path("tools/tool02_scene_requirement_generator").exists()
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


def test_failed_validation_can_stop_when_editor_backend_is_unavailable(tmp_path):
    agent, client = make_agent(tmp_path, [
        call("scene_validator", {"rendered_video": "/synthetic/old.mp4", "scene_requirement": "A grounded vehicle.", "mock_force_fail": True}),
        {"action": "stop", "reason": "Validation failed; the new editor backend is not configured to repair it."},
    ])
    state = agent.run("Validate the supplied scene and repair it if possible.")
    assert state["status"] == "error"
    feedback = json.loads(client.messages[1][1]["content"])
    assert feedback["current_state"]["history"][-1]["result"]["result"]["passed"] is False
    assert "scene_editor" not in {t["name"] for t in feedback["available_tools"]}


def test_repeated_transient_failure_stops_as_error_instead_of_success(tmp_path):
    arguments={"requirement":"a dataset scene"}
    decisions=[
        call("video_retriever",arguments),
        call("video_retriever",arguments),
        {"action":"stop","reason":"Retrieval timed out again after one retry; no supported alternative is available."},
    ]
    agent,client=make_agent(tmp_path,decisions)
    error={"status":"error","tool":"video_retriever","error":{"type":"timeout","message":"Transient retrieval timeout"}}
    with patch.object(agent.executor,"execute",return_value=error) as execute:
        state=agent.run("Retrieve a bound dataset scene.")
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
        call("video_retriever",{"requirement":"a suitable dataset scene"}),
    ])
    planner=Orchestrator(llm_client=client)
    result=planner.decide("Repair the failed edit.",{"history":[]})
    assert result["tool_name"]=="video_retriever"
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
    result=Orchestrator(llm_client=client).decide("Simulate the scene.",{"history":[]})
    assert result["action"]=="stop"
    assert "checkpoint" in result["reason"]
    assert "stop format" in client.messages[-1][-1]["content"]


def test_main_agent_passes_merged_candidate_directly_to_tool04(tmp_path, monkeypatch):
    from tools.tool02_scene_instance_generator import real as tool02
    monkeypatch.setattr(tool02, 'RECORDS_ROOT', tmp_path / 'tool02_records')
    analysis={'name':'Cross traffic', 'wrong_behavior':'Late yielding to cross traffic.'}
    class RequirementFlowClient(DecisionClient):
        def chat(self, messages):
            self.messages.append(deepcopy(messages))
            payload=json.loads(messages[1]['content'])
            if len(self.messages)==1:
                return json.dumps(call('scene_instance_generator', {'failure_analysis':analysis}))
            if len(self.messages)==2:
                previous=payload['current_state']['history'][-1]['result']
                return json.dumps(call('video_retriever', {'requirement':previous['result']['candidates'][0]['base_video_requirement']}))
            return json.dumps({'action':'finish','reason':'The requested scenario requirement and conception are available.'})
    client=RequirementFlowClient([])
    orchestrator=Orchestrator(llm_client=client)
    config=replace(AgentConfig.load(),tool_modes={name:'mock' for name in AgentConfig.load().tool_modes},logs_root=tmp_path/'agent')
    with patch('agent.main.Orchestrator',return_value=orchestrator):
        agent=Agent(config)
    state=agent.run('Use the supplied analysis to generate candidates and retrieve the first base video; no editing.')
    results=[entry for entry in state['history'] if 'tool' in entry]
    assert state['status']=='finished'
    assert [entry['tool'] for entry in results]==['scene_instance_generator','video_retriever']
    first=results[0]['result']['result']
    assert set(first)=={'scene_requirement','target_effect','candidates'}
    assert results[1]['result']['result']['requirement']==first['candidates'][0]['base_video_requirement']
    next_context=json.loads(client.messages[1][1]['content'])
    assert next_context['current_state']['current_stage']=='requirement_generation'
    names={tool['name'] for tool in next_context['available_tools']}
    assert 'scene_instance_generator' in names
    assert 'scene_conceiver' not in names
    assert 'scene_requirement_generator' not in names


def test_new_tool02_default_real_dispatch_and_old_name_is_unavailable(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.tool02_scene_instance_generator import real as tool02
    monkeypatch.setattr(tool02,'RECORDS_ROOT',tmp_path/'tool02_records')
    from tools.tool02_scene_instance_generator.mock import run as mock_scene
    response=mock_scene({'failure_analysis':{'failure_reason':'Delayed yielding'}})['result']
    client=SimpleNamespace(model='offline-injected',chat=lambda messages:json.dumps(response))
    out=call_tool('scene_instance_generator',{'failure_analysis':{'failure_reason':'Delayed yielding'},'_llm_client':client})
    assert out['status']=='success'
    assert out['meta']['mode']=='real'
    assert out['result']==response
    assert AgentConfig.load().tool_modes['scene_instance_generator']=='real'
    with pytest.raises(ValueError,match='Unknown tool'):
        call_tool('scene_requirement_generator',{},mode='mock')
