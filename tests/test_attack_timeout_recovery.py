"""Stream deadlines retain evidence and permit bounded saved-candidate review only."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
if str(CORE) not in sys.path:sys.path.insert(0,str(CORE))

from streaming_requests import (StreamingRequestArchive, StreamingRequestFailure,
                               RequestDeadlineExceeded, request_protocol_details, request_timeout_details)
from attack_budget import BudgetedArchive
import run_attack_workflow as workflow


def deadline_failure(kind='stream_wall_time_exceeded'):
    return StreamingRequestFailure('fixture',failure_kind=kind,error_type='TimeoutError',
        phase='reading_response_stream',archive_ref='/offline/attempt',output_received=True,chunks_received=12)


def protocol_failure():
    return StreamingRequestFailure('fixture',failure_kind='multiple_tool_calls_when_disabled',
        error_type='ValueError',phase='reading_response_stream',archive_ref='/offline/protocol',
        output_received=True,chunks_received=20)


def test_protocol_failure_is_distinct_from_timeout():
    assert request_protocol_details(protocol_failure())['failure_kind']=='multiple_tool_calls_when_disabled'
    assert request_timeout_details(protocol_failure()) is None
    assert request_protocol_details(deadline_failure()) is None


@pytest.mark.parametrize('kind',['stream_wall_time_exceeded','request_wall_time_exceeded'])
def test_both_wall_deadlines_are_model_decision_timeouts_even_when_wrapped(kind):
    inner=deadline_failure(kind)
    outer=RuntimeError('outer wrapper');outer.__cause__=inner
    details=request_timeout_details(outer)
    assert details['timeout_category']=='model_decision_timeout'
    assert details['failure_kind']==kind and details['output_received'] is True
    assert details['archive_ref']=='/offline/attempt'
    assert request_timeout_details(ConnectionError('connection closed')) is None
    assert request_timeout_details(ValueError('invalid schema')) is None


def test_deadline_after_reasoning_is_archived_without_replaying_partial_call(tmp_path,monkeypatch):
    import streaming_requests
    calls=[]
    def interrupted(*args,**kwargs):
        calls.append(1)
        yield 'headers',None
        yield 'chunk',NS(model='offline',id='1',usage=None,choices=[NS(
            delta=NS(reasoning_content='reasoning',content=None,tool_calls=None),finish_reason=None)])
        raise RequestDeadlineExceeded('deadline')
    monkeypatch.setattr(streaming_requests,'deadline_events',interrupted)
    archive=StreamingRequestArchive(tmp_path,NS(),dict(api_key='',base_url='https://example.invalid',model='offline',timeout=300))
    with pytest.raises(StreamingRequestFailure) as caught:
        archive.complete('edit',[dict(role='user',content='fixture')],retries=2)
    assert len(calls)==1
    assert request_timeout_details(caught.value)['timeout_category']=='model_decision_timeout'
    attempt=Path(caught.value.archive_ref)
    assert json.loads((attempt/'error.json').read_text())['timeout_category']=='model_decision_timeout'
    assert json.loads((attempt/'partial-response.json').read_text())['reasoning_content']=='reasoning'
    assert not (attempt/'response.json').exists()


@pytest.fixture
def editor(tmp_path,monkeypatch):
    import decision_scene_view
    monkeypatch.setattr(decision_scene_view,'prepare_attack_scene',lambda *args:None)
    value=NS(run_dir=tmp_path,constraints={},history=[],nodes={'seed_000':{'result':{'reference_screen_passed':True}}},
        pdm_probe=None,policy_probe=None,inspected={'seed_000'},submitted=None,target={'track_id':'target'})
    value.public=lambda key:dict(candidate_id=key,reference_screen_passed=value.nodes[key]['result']['reference_screen_passed'])
    value.context=lambda:dict(constructor_budget={'calls':0,'evaluations':0,'remaining_evaluations':10},
        candidates=[value.public(key) for key in value.nodes])
    def submit(key,reason):
        value.submitted=key
        return dict(status='accepted',candidate_id=key)
    value.submit=submit
    return value


def execute(editor,archive):
    return workflow.edit_object(editor,{'instruction':'fixture'},archive,300,16384,4)


@pytest.mark.parametrize('time_at_timeout',[419.99,420.01])
def test_timeout_recovers_once_with_review_tools_and_unchanged_thinking(editor,monkeypatch,time_at_timeout):
    requests=[]
    now=[0.]
    monkeypatch.setattr(workflow,'perf_counter',lambda:now[0])
    class Archive:
        def complete(self,label,messages,**kwargs):
            requests.append((messages,kwargs))
            if len(requests)==1:
                now[0]=time_at_timeout
                raise deadline_failure()
            return '',Path('/offline/complete')
    monkeypatch.setattr(workflow,'decode_call',lambda path:({'id':'fixture','function':{'name':'submit_candidate'}},
        'submit_candidate',{'candidate_id':'seed_000','reason':'reviewed saved candidate'}))
    assert execute(editor,Archive())=='seed_000'
    assert len(requests)==2
    first,second=(row[1] for row in requests)
    assert second['total_seconds']==90
    assert second['reasoning_effort']==first['reasoning_effort']
    assert second['max_tokens']==first['max_tokens']
    assert {tool['function']['name'] for tool in second['tools']}<={'inspect_trajectory','submit_candidate','finish_search','probe_vadv2','probe_pdm'}
    assert 'one bounded review' in requests[1][0][0]['content']
    output=json.loads((editor.run_dir/'output.json').read_text())
    assert output['finalization_grace_used'] is True
    assert output['request_timeout_recovery']['timeout_category']=='model_decision_timeout'


@pytest.mark.parametrize('qualified',[False,True])
def test_single_request_deadline_keeps_editing_within_original_remaining_stage(editor,monkeypatch,qualified):
    editor.nodes['seed_000']['result']['reference_screen_passed']=qualified
    now=[0.];requests=[]
    monkeypatch.setattr(workflow,'perf_counter',lambda:now[0])
    class Archive:
        def complete(self,label,messages,**kwargs):
            requests.append((messages,kwargs))
            if len(requests)==1:
                now[0]=300.
                raise deadline_failure()
            return '',Path('/offline/complete')
    monkeypatch.setattr(workflow,'decode_call',lambda path:({'id':'finish','function':{'name':'finish_search'}},
        'finish_search',{'reason':'fixture ends after checking recovery tool access'}))
    assert execute(editor,Archive()) is None
    assert len(requests)==2
    first,second=(row[1] for row in requests)
    assert second['total_seconds']==120  # original 420s stage, not a fresh 300s request
    assert second['reasoning_effort']==first['reasoning_effort'] and second['max_tokens']==first['max_tokens']
    assert 'edit_speed_profile' in {tool['function']['name'] for tool in second['tools']}
    assert 'continue_editing' in requests[1][0][0]['content']
    assert '"finalization_only":false' in requests[1][0][0]['content']
    output=json.loads((editor.run_dir/'output.json').read_text())
    assert output['editing_timeout_recovery_used'] is True
    assert output['finalization_grace_used'] is False


@pytest.mark.parametrize('qualified,expected_calls',[(False,2),(True,3)])
def test_repeated_timeout_does_not_loop_or_execute_partial_actions(editor,monkeypatch,qualified,expected_calls):
    editor.nodes['seed_000']['result']['reference_screen_passed']=qualified
    calls=[];now=[0.]
    monkeypatch.setattr(workflow,'perf_counter',lambda:now[0])
    class Archive:
        def complete(self,*args,**kwargs):
            calls.append(kwargs['total_seconds'])
            now[0]+=kwargs['total_seconds']
            raise deadline_failure()
    monkeypatch.setattr(workflow,'decode_call',lambda path:pytest.fail('must not decode partial response'))
    assert execute(editor,Archive()) is None
    assert len(calls)==expected_calls and editor.submitted is None
    output=json.loads((editor.run_dir/'output.json').read_text())
    assert output['termination_evidence']['timeout_category']=='model_decision_timeout'
    assert output['resumable_candidate_ids']==(['seed_000'] if qualified else [])
    assert output['stop_reason']=='stage_timeout_exhausted'
    assert calls==([300,120,90] if qualified else [300,120])


def test_exhausted_search_stops_before_api_but_reserve_can_review_a_saved_candidate(editor,monkeypatch):
    monkeypatch.setenv('ADVERSARIAL_ATTACK_SESSION',str(editor.run_dir/'session'))
    requests=[]
    class Archive:
        def complete(self,*args,**kwargs):
            requests.append(kwargs)
            return '',Path('/offline/complete')
    budget=BudgetedArchive(Archive(),editor.run_dir)
    budget.path.parent.mkdir(parents=True)
    budget.path.write_text(json.dumps(dict(requests=3,api_seconds=810.)))
    editor.nodes['seed_000']['result']['reference_screen_passed']=False
    assert execute(editor,budget) is None
    assert not requests
    assert json.loads((editor.run_dir/'output.json').read_text())['stop_reason']=='cumulative_attack_api_budget_exhausted'
    editor.nodes['seed_000']['result']['reference_screen_passed']=True
    monkeypatch.setattr(workflow,'decode_call',lambda path:({'id':'fixture','function':{'name':'submit_candidate'}},
        'submit_candidate',{'candidate_id':'seed_000','reason':'existing checked candidate'}))
    assert execute(editor,budget)=='seed_000'
    assert len(requests)==1 and requests[0]['total_seconds']<=90
    assert budget.context()['requests']==4


def test_multiple_tool_calls_get_one_correction_without_executing_partial_calls(editor,monkeypatch):
    requests=[]
    class Archive:
        def complete(self,label,messages,**kwargs):
            requests.append((messages,kwargs))
            if len(requests)==1:raise protocol_failure()
            return '',Path('/offline/complete')
    monkeypatch.setattr(workflow,'decode_call',lambda path:({'id':'finish','function':{'name':'finish_search'}},
        'finish_search',{'reason':'corrected to one call'}))
    assert execute(editor,Archive()) is None
    assert len(requests)==2
    text=requests[1][0][0]['content']
    assert 'multiple tool calls' in text and 'No tool was executed' in text
    assert 'single_tool_call_correction' in text
    output=json.loads((editor.run_dir/'output.json').read_text())
    assert output['request_protocol_recovery']['failure_kind']=='multiple_tool_calls_when_disabled'


def test_multiple_tool_call_recovery_is_bounded(editor):
    calls=[]
    class Archive:
        def complete(self,*args,**kwargs):
            calls.append(1);raise protocol_failure()
    with pytest.raises(StreamingRequestFailure):
        execute(editor,Archive())
    assert len(calls)==2


def test_repeated_evaluation_returns_cached_evidence(editor,monkeypatch):
    editor.constraints['agent_decides_interaction']=True
    editor.evaluate_candidate=editor.public
    calls=[('evaluate_candidate',{'candidate_id':'seed_000'}),
        ('evaluate_candidate',{'candidate_id':'seed_000'}),('finish_search',{'reason':'cached evidence reviewed'})]
    requests=[]
    class Archive:
        def complete(self,label,messages,**kwargs):
            requests.append(messages);return '',Path('/offline/complete')
    def decode(path):
        name,args=calls.pop(0)
        return {'id':'fixture','function':{'name':name}},name,args
    monkeypatch.setattr(workflow,'decode_call',decode)
    assert execute(editor,Archive()) is None
    assert len(requests)==3
    cached=json.loads((editor.run_dir/'tool_calls/turn_02/result.json').read_text())
    assert cached['cached'] is True and cached['cache_scope']=='previous_evaluation'


def test_repeated_inspection_reuses_image_without_redelivery(editor,monkeypatch):
    editor.constraints['agent_decides_interaction']=True
    monkeypatch.setattr(workflow,'image_part',lambda path:dict(type='image_url',path=path))
    count=[0]
    def inspect(candidate):
        count[0]+=1
        return dict(status='inspection_ready',candidate_id=candidate,path='/offline/candidate.png',
            metrics=editor.public(candidate),cached=count[0]>1)
    editor.inspect=inspect
    calls=[('inspect_trajectory',{'candidate_id':'seed_000'}),
        ('inspect_trajectory',{'candidate_id':'seed_000'}),('finish_search',{'reason':'cached image reviewed'})]
    requests=[]
    class Archive:
        def complete(self,label,messages,**kwargs):
            requests.append(messages);return '',Path('/offline/complete')
    def decode(path):
        name,args=calls.pop(0)
        return {'id':'fixture','function':{'name':name}},name,args
    monkeypatch.setattr(workflow,'decode_call',decode)
    assert execute(editor,Archive()) is None
    assert count[0]==2 and len(requests)==3
    cached=json.loads((editor.run_dir/'tool_calls/turn_02/result.json').read_text())
    assert cached['cached'] is True
    assert isinstance(requests[2][0]['content'],str)
