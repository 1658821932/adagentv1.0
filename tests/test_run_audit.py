"""Offline integration checks for API archives, trace propagation and reports."""
import base64
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from agent.llm_client import LLMClient
from agent.logger import RunLogger
from agent.orchestrator import Orchestrator
from common.llm_audit import RequestAudit, audited_json_call, record_tool_exchange, sum_usage
from common.run_report import summarize
from mcp_servers.runtime import Run, trace_context

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT/'mcp_servers/scene_editing/core/adversarial'


def records(root):
    return [json.loads(p.read_text()) for p in (root/'api_requests').glob('*/result.json')]


def test_request_payload_is_preserved_without_credentials_or_private_continuation(tmp_path, monkeypatch):
    secret = 'fixture-key-for-audit-only'
    monkeypatch.setenv('API_KEY_99', secret)
    image = 'data:image/png;base64,' + base64.b64encode(b'fixture image').decode()
    payload = dict(model='fixture', messages=[
        dict(role='system', content='Prompt '+secret),
        dict(role='assistant', content='public answer', reasoning_content='opaque-provider-data'),
        dict(role='user', content=[dict(type='image_url', image_url=dict(url=image))])],
        tools=[dict(type='function', function=dict(name='edit', parameters=dict(type='object')))],
        extra_body=dict(thinking=dict(type='enabled')), reasoning_effort='high')
    before = deepcopy(payload)
    with trace_context(agent_log_dir=str(tmp_path), agent_run_id='run_fixture', stage='main_agent', step=0):
        for _ in range(2):
            audit = RequestAudit(payload, endpoint='https://example.invalid/v1?api_key='+secret)
            audit.first_response()
            audit.finish(response=dict(content='public', reasoning_content='opaque-provider-output'), usage=dict(prompt_tokens=42))
    assert payload == before
    assert len(list((tmp_path/'assets').glob('*.bin'))) == 1
    text = '\n'.join(p.read_text() for p in tmp_path.rglob('*.json'))
    assert secret not in text and 'opaque-provider-data' not in text and 'opaque-provider-output' not in text
    request = json.loads(next((tmp_path/'api_requests').glob('*/request.json')).read_text())
    assert request['payload']['extra_body'] == payload['extra_body']
    assert request['payload']['tools'] == payload['tools']
    assert request['payload']['messages'][2]['content'][0]['image_url']['url']['bytes'] == 13
    assert records(tmp_path)[0]['input_metrics']['image_count'] == 1
    assert records(tmp_path)[0]['usage']['prompt_tokens'] == 42


def test_transport_error_and_non_json_http_error_are_recorded(tmp_path):
    with trace_context(agent_log_dir=str(tmp_path)):
        with pytest.raises(TimeoutError):
            audited_json_call(lambda: (_ for _ in ()).throw(TimeoutError('do not archive raw error')), dict(model='fixture'))
        response = Mock(spec=['status_code', 'json'], status_code=502)
        response.json.side_effect = ValueError('upstream returned HTML')
        with pytest.raises(ValueError):
            audited_json_call(lambda: response, dict(model='fixture'))
    rows = records(tmp_path)
    assert len(rows) == 2 and all(r['status'] == 'error' for r in rows)
    assert {r['error_type'] for r in rows} == {'TimeoutError', 'ValueError'}
    assert next(r for r in rows if r['error_type']=='ValueError')['http_status'] == 502
    assert 'do not archive raw error' not in '\n'.join(p.read_text() for p in tmp_path.rglob('*.json'))


class Stream:
    def __init__(self, name='finish', arguments='{"reason":"complete"}'):
        self.closed = False
        self.chunk = NS(model='fixture', id='fixture', usage=None, choices=[NS(index=0,
            delta=NS(content=None, reasoning_content=None, tool_calls=[NS(index=0, id='call',
                type='function', function=NS(name=name, arguments=arguments))]), finish_reason='tool_calls')])
    def __iter__(self):
        yield self.chunk
        yield NS(model='fixture', id='fixture', choices=[], usage=NS(model_dump=lambda:dict(prompt_tokens=10, completion_tokens=2)))
    def close(self):
        self.closed = True
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


def client(streams):
    result = LLMClient.__new__(LLMClient)
    result.model = 'fixture'
    result.base_url = 'https://example.invalid/v1'
    result.max_tokens = 1024
    result.on_delta = None
    result.client = Mock()
    result.client.chat.completions.create.side_effect = streams
    return result


def test_main_correction_saves_both_actual_contexts_and_inclusive_usage(tmp_path):
    logger = RunLogger('offline fixture', 2, 'fixture', 'https://example.invalid/v1', {}, logs_root=tmp_path)
    llm = client([Stream('unknown_tool'), Stream()])
    with trace_context(agent_log_dir=str(logger.run_dir), agent_run_id=logger.run_id, stage='main_agent', step=0):
        decision = Orchestrator(llm).decide('offline fixture', {})
    assert decision['action'] == 'finish'
    logger.log_decision(0, decision, dict(metrics=llm.last_request_metrics, usage=llm.last_usage))
    logger.finalize(dict(status='finished', current_step=0))
    rows = sorted(records(logger.run_dir), key=lambda r:r['context']['decision_attempt'])
    assert len(rows) == 2 and rows[1]['context']['decision_attempt'] == 2
    request = json.loads((logger.run_dir/'api_requests'/rows[1]['request_id']/'request.json').read_text())
    assert any('validation_errors' in str(m) for m in request['payload']['messages'])
    summary = json.loads((logger.run_dir/'summary.json').read_text())
    assert summary['api_calls'] == 2
    assert summary['stages'][0]['main_correction_requests'] == 1
    event = json.loads(logger.agent_log_path.read_text())
    assert event['llm_request']['usage']['prompt_tokens'] == 20
    assert event['llm_request']['metrics']['request_count'] == 2
    assert (logger.run_dir/'snapshot/files/agent/prompt.md').is_file()
    assert (logger.run_dir/'REPORT.md').is_file()


def test_failure_before_stream_headers_is_not_lost(tmp_path):
    llm = client([TimeoutError('fixture timeout')])
    with trace_context(agent_log_dir=str(tmp_path), stage='main_agent'):
        with pytest.raises(TimeoutError):
            llm.call_functions([], [])
    row = records(tmp_path)[0]
    assert row['status'] == 'error' and row['error_type'] == 'TimeoutError'
    assert row['elapsed_seconds'] >= 0
    assert llm.last_request_metrics['elapsed_seconds'] >= 0


def test_service_trace_survives_real_subprocess_and_threads(tmp_path):
    with trace_context(agent_log_dir=str(tmp_path/'agent'), agent_run_id='fixture',
                       tool_call_id='fixture:step_01:failure_analyzer', stage='failure_analyzer', step=1):
        service = Run('fixture_service', 'fixture', directory=tmp_path/'service')
        env = service.child_env()
    script = '''from concurrent.futures import ThreadPoolExecutor
from common.llm_audit import audited_json_call
def work(i):
    return audited_json_call(lambda: {'usage': {'prompt_tokens': i}}, {'model':'fixture','messages':[]})
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(work, range(8)))
'''
    subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=env, check=True, capture_output=True)
    rows = records(tmp_path/'agent')
    assert len(rows) == 8
    assert all(r['stage']=='failure_analyzer' and r['context']['agent_run_id']=='fixture' for r in rows)


def test_edit_events_are_deduplicated_and_operation_times_are_not_added(tmp_path):
    with trace_context(agent_log_dir=str(tmp_path), tool_call_id='fixture:step_05:ego_trajectory_editor'):
        call = dict(function=dict(name='edit_speed_profile', arguments='{"reason":"fixture"}'))
        for _ in range(2):
            record_tool_exchange(call, dict(status='invalid'), tmp_path/'turn_01')
        audit = RequestAudit(dict(model='fixture'))
        audit.finish()
    request = next((tmp_path/'api_requests').glob('*/result.json'))
    row = json.loads(request.read_text()); row['elapsed_seconds'] = 3
    request.write_text(json.dumps(row))
    tool = tmp_path/'tools/step_05_ego_trajectory_editor'; tool.mkdir(parents=True)
    (tool/'call.json').write_text(json.dumps(dict(tool='ego_trajectory_editor')))
    (tool/'timing.json').write_text(json.dumps(dict(elapsed_seconds=10)))
    report = summarize(tmp_path)
    assert report['api_seconds'] == 3
    stage = report['stages'][0]
    assert stage['tool_wall_seconds'] == 10 and stage['edit_attempts'] == 1
    assert stage['subagent_tool_calls'] == 1


def test_existing_editor_stream_archive_also_links_to_main_run(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(CORE))
    monkeypatch.setenv('ADVERSARIAL_HOST_HOOKS', 'mcp_servers.scene_editing.adapter.core_hooks')
    module = importlib.import_module('streaming_requests')
    sdk = Mock()
    stream = Stream()
    sdk.chat.completions.create.return_value = stream
    profile = dict(model='fixture', base_url='https://example.invalid/v1', api_key='fixture-key-for-editor', timeout=10)
    archive = module.StreamingRequestArchive(tmp_path/'editor/requests', sdk, profile)
    with trace_context(agent_log_dir=str(tmp_path/'agent'), tool_call_id='fixture:step_05:ego_trajectory_editor'):
        _, attempt = archive.complete('turn_01', [dict(role='user', content='fixture')],
            tools=[dict(type='function', function=dict(name='finish'))], allow_tool_calls=True)
    row = records(tmp_path/'agent')[0]
    assert row['status']=='success' and row['archive_ref']==str(attempt)
    assert row['usage']['prompt_tokens']==10
    assert row['stage']=='ego_trajectory_editor' and stream.closed


def test_disabled_audit_never_writes_to_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv('ADAGENT_AGENT_LOG_DIR', raising=False)
    monkeypatch.chdir(tmp_path)
    assert audited_json_call(lambda: {'answer':1}, dict(model='fixture')) == {'answer':1}
    assert not list(tmp_path.iterdir())


def test_sdk_model_with_json_method_is_returned_unchanged(tmp_path):
    class SDKResponse:
        usage = {'prompt_tokens': 9}
        def model_dump(self):
            return dict(usage=self.usage, choices=[dict(message=dict(content='public fixture'))])
        def json(self):
            return json.dumps(self.model_dump())
    response = SDKResponse()
    with trace_context(agent_log_dir=str(tmp_path)):
        actual = audited_json_call(lambda: response, dict(model='fixture'))
    assert actual is response
    assert records(tmp_path)[0]['usage']['prompt_tokens'] == 9


def test_correction_attempts_and_nested_usage_are_inclusive(tmp_path):
    with trace_context(agent_log_dir=str(tmp_path), stage='edit_planner', llm_attempt=2):
        audited_json_call(lambda: dict(usage=dict(prompt_tokens=12)), dict(model='fixture'))
    with trace_context(agent_log_dir=str(tmp_path), stage='video_retriever'):
        audited_json_call(lambda: {}, dict(model='fixture'), label='embedding/attempt_3')
    assert sorted(r['attempt_index'] for r in records(tmp_path)) == [2, 3]
    assert sum(r['retry_requests'] for r in summarize(tmp_path)['stages']) == 2
    assert sum_usage([dict(prompt_tokens=3, prompt_tokens_details=dict(cached_tokens=2)),
                      dict(prompt_tokens=4, prompt_tokens_details=dict(cached_tokens=3))]) == dict(
                          prompt_tokens=7, prompt_tokens_details=dict(cached_tokens=5))


def test_resumed_editor_request_is_counted_as_retry_and_context_peaks_are_reported(tmp_path):
    with trace_context(agent_log_dir=str(tmp_path), stage='ego_trajectory_editor'):
        first = RequestAudit(dict(model='fixture', messages=[dict(role='user', content='fixture')]),
                             logical_key='editor/turn_01')
        first.finish(error=TimeoutError('fixture'))
        second = RequestAudit(dict(model='fixture', messages=[dict(role='user', content='fixture')]),
                              logical_key='editor/turn_01')
        second.finish(usage=dict(prompt_tokens=35))
    assert sorted(r['attempt_index'] for r in records(tmp_path)) == [1, 2]
    stage = summarize(tmp_path)['stages'][0]
    assert stage['retry_requests']==1 and stage['api_errors']==1
    assert stage['max_prompt_tokens']==35 and stage['max_input_text_characters']>=7


def test_failed_tool_timing_is_preserved_at_finalize(tmp_path):
    logger = RunLogger('fixture', 1, 'fixture', 'https://example.invalid', {}, logs_root=tmp_path)
    logger.start_tool_call(0, 'fixture_tool', {})
    logger.log_tool_error(0, 'fixture_tool', TimeoutError('fixture'))
    logger.finalize(dict(status='error', current_step=0))
    timing = json.loads((logger.run_dir/'tools/step_00_fixture_tool/timing.json').read_text())
    assert timing['status'] == 'error' and timing['elapsed_seconds'] >= 0


def test_failure_analysis_retries_are_archived_without_real_network(tmp_path, monkeypatch):
    scripts = ROOT/'mcp_servers/worldengine_failure_analysis/core/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    review = importlib.import_module('vlm_review')
    class Reply:
        def __enter__(self):
            from io import StringIO
            return StringIO(json.dumps(dict(choices=[dict(finish_reason='stop', message=dict(content='{"action":"fixture"}'))],
                                           usage=dict(prompt_tokens=4))))
        def __exit__(self, *args):
            pass
    open_request = Mock(side_effect=[TimeoutError('fixture'), Reply()])
    monkeypatch.setattr(review.urllib.request, 'urlopen', open_request)
    monkeypatch.setattr(review.time, 'sleep', lambda _:None)
    secret = dict(model='fixture', endpoint='https://example.invalid/chat/completions',
                  api_key='fixture-wfa-key', max_tokens=128, timeout=2)
    with trace_context(agent_log_dir=str(tmp_path), stage='failure_analyzer'):
        result = review.Client(secret, tmp_path, audit_factory=RequestAudit).ask(
            [dict(role='user', content='fixture')], 'turn_01')
    assert result['action']=='fixture' and open_request.call_count==2
    rows = records(tmp_path)
    assert len(rows)==2 and sorted(r['status'] for r in rows)==['error','success']
    assert sorted(r['attempt_index'] for r in rows)==[1,2]
