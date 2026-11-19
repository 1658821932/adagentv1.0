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

    def call_functions(self,messages,tools):
        self.functions=tools
        decision=json.loads(self.chat(messages))
        name=decision["tool_name"] if decision["action"]=="call_tool" else decision["action"]
        payload={"reason":decision["reason"]}
        if decision["action"]=="call_tool":
            payload["arguments"]=decision["arguments"]
        return {"message":{"role":"assistant","content":None,"tool_calls":[{
            "id":"test_call","type":"function","function":{"name":name,"arguments":json.dumps(payload)}}]},
            "finish_reason":"tool_calls"}

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
    assert "scene_validator" not in names
    assert "edit_planner" in names
    assert "scene_replayer" not in names
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
    assert "finish/stop" in client.messages[-1][-1]["content"]


def test_main_agent_passes_merged_candidate_directly_to_tool03(tmp_path, monkeypatch):
    from tools.tool02_scene_instance_generator import real as tool02
    monkeypatch.setattr(tool02, 'RECORDS_ROOT', tmp_path / 'tool02_records')
    analysis={'failure':{'outcome':'collision','interaction':'crossing',
                         'ego_cause':'Late yielding to cross traffic.'},
              'violation_step':8,
              'attack':{'map_context':'t_junction','ego_action':'turn_right','ego_lane_keeping':'keep',
                        'other_origin':'right_arm','other_action':'turn_left','other_lane_keeping':'keep'},
              'environment':{'time':'2021-09-16T20:16:41+00:00','weather':'cloudy','extra':'dry'},
              'verification':{'sample_frames':[6,8,10],'corrections':[],'notes':[]}}
    class RequirementFlowClient(DecisionClient):
        def chat(self, messages):
            self.messages.append(deepcopy(messages))
            payload=json.loads(messages[1]['content'])
            if len(self.messages)==1:
                return json.dumps(call('scene_instance_generator', {'failure_analysis':analysis}))
            if len(self.messages)==2:
                previous=payload['current_state']['history'][-1]['result']
                candidate=previous['result']['candidates'][0]
                return json.dumps(call('video_retriever', {'requirement':candidate['retrieval_query'],
                    'candidate_id':candidate['candidate_id']}))
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
    assert set(first)=={'target_scene','attack','candidates'}
    assert results[1]['result']['result']['requirement']==first['candidates'][0]['retrieval_query']
    assert results[1]['result']['result']['candidate_id']==first['candidates'][0]['candidate_id']
    next_context=json.loads(client.messages[1][1]['content'])
    assert next_context['current_state']['current_stage']=='requirement_generation'
    names={tool['function']['name'] for tool in client.functions}
    assert 'scene_instance_generator' in names
    assert 'scene_conceiver' not in names
    assert 'scene_requirement_generator' not in names


def test_main_prompt_uses_current_tool02_to_tool04_handoff():
    prompt = Path('agent/prompt.md').read_text()
    assert 'scenario_result' in prompt and 'retrieval_result' in prompt
    assert 'paired edit_strategy' not in prompt
    assert 'scene_requirement, target_effect and candidates' not in prompt


def test_new_tool02_default_real_dispatch_and_old_name_is_unavailable(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.tool02_scene_instance_generator import real as tool02
    monkeypatch.setattr(tool02,'RECORDS_ROOT',tmp_path/'tool02_records')
    from tools.tool02_scene_instance_generator.mock import run as mock_scene
    sample={'failure':{'outcome':'collision','interaction':'crossing','ego_cause':'Delayed yielding too late.'},
            'violation_step':8,
            'attack':{'map_context':'t_junction','ego_action':'turn_right','ego_lane_keeping':'keep',
                      'other_origin':'right_arm','other_action':'turn_left','other_lane_keeping':'keep'},
            'environment':{'time':'2021-09-16T20:16:41+00:00','weather':'cloudy','extra':'dry'},
            'verification':{'sample_frames':[6,8,10],'corrections':[],'notes':[]}}
    response=mock_scene({'failure_analysis':sample})['result']
    client=SimpleNamespace(model='offline-injected',chat=lambda messages:json.dumps(response))
    out=call_tool('scene_instance_generator',{'failure_analysis':sample,'_llm_client':client})
    assert out['status']=='success'
    assert out['meta']['mode']=='real'
    assert out['result']==response
    # Runtime modes are user-editable; the default dispatcher is the contract.
    from tools.registry import DEFAULT_TOOL_MODES
    assert DEFAULT_TOOL_MODES['scene_instance_generator']=='real'
    with pytest.raises(ValueError,match='Unknown tool'):
        call_tool('scene_requirement_generator',{},mode='mock')


def test_running_job_is_waited_without_another_llm_decision(tmp_path):
    agent,client=make_agent(tmp_path,[
        call("failure_analyzer",{"token":"example"}),
        {"action":"finish","reason":"Analysis complete."},
    ])
    pending={"status":"running","tool":"failure_analyzer","job":{"job_id":"a"*32,"state":"running"}}
    done={"status":"success","tool":"failure_analyzer","result":{"failure_analysis":{}}}
    with patch.object(agent.executor,"execute",return_value=pending) as execute, patch.object(
            agent.executor,"query",return_value=done) as query:
        result=agent.run("Analyze one scene")
    assert result["status"]=="finished"
    execute.assert_called_once()
    query.assert_called_once()
    assert len(client.messages)==2
    assert result["pending_job"] is None


def test_agent_records_orphaned_job_and_ends_without_retry(tmp_path):
    agent, client = make_agent(tmp_path, [call('failure_analyzer', {'token': 'example'})])
    job = dict(job_id='a'*32, state='running', manifest_path='/fixture/manifest.json')
    pending = dict(status='running', tool='failure_analyzer', job=job)
    orphan = dict(pending, job=dict(job, state='orphaned'))
    with patch.object(agent.executor, 'execute', return_value=pending) as execute, patch.object(
            agent.executor, 'query', return_value=orphan) as query:
        result = agent.run('Analyze one scene')
    execute.assert_called_once()
    query.assert_called_once()
    assert len(client.messages) == 1
    assert result['status'] == 'error' and result['pending_job'] is None
    recorded = next(item['result'] for item in result['history'] if item.get('tool') == 'failure_analyzer')
    assert recorded['error']['type'] == 'job_orphaned'
    assert recorded['job']['manifest_path'] == job['manifest_path']
