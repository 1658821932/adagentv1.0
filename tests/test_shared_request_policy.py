"""Host/editor policy equivalence, native phase gates and opaque continuation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import api_config
from agent.llm_client import LLMClient
from agent.orchestrator import Orchestrator
from mcp_servers.scene_editing.adapter import service
from mcp_servers.scene_editing.core.adversarial.llm_request_policy import apply_request_policy,validate_request_policy
from mcp_servers.scene_editing.core.adversarial.ego_tool_transport import require_native_call,validate_exposed_call

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
CASES=[('thinking','max'),('kimi','high'),('qwen','xhigh')]

def profile(dialect,effort):
    return dict(request_dialect=dialect,thinking_mode='enabled',reasoning_effort=effort,
        tool_choice_policy='auto',max_tokens=65536,timeout=600,model='fixture-model',
        base_url='https://example.invalid',api_key_env='FAKE_TEST_KEY')

def tools():
    return [dict(type='function',function=dict(name=n,parameters=dict(type='object',properties=dict(candidate_id=dict(type='string',enum=['a'])),required=['candidate_id'],additionalProperties=False))) for n in ['inspect','submit']]

@pytest.mark.parametrize('dialect,effort',CASES)
def test_host_and_backend_have_identical_managed_parameters(monkeypatch,dialect,effort):
    p=profile(dialect,effort);p['api_key']='fake-only-fixture'
    monkeypatch.setattr('common.api_profiles.reasoning_effort_level',lambda:None)
    # This equivalence test uses the fixture profile, independent of local UI pins.
    monkeypatch.setattr('common.api_profiles.effort_for',lambda component:None)
    monkeypatch.setattr(service,'get_profile',lambda *a,**kw:p)
    env=service.backend_api_env(['python',str(CORE/'editing_backend.py')],{})
    for k,v in env.items():monkeypatch.setenv(k,v)
    from mcp_servers.scene_editing.core.adversarial.kimi_request_parameters import normalize_kimi_parameters
    kwargs=dict(model=p['model'],max_tokens=8192,reasoning_effort='low',extra_body=dict(thinking=dict(type='disabled')),tools=tools(),tool_choice='required')
    host=LLMClient.__new__(LLMClient);host.request_profile=p
    actual=host._request_parameters(kwargs)
    assert actual==normalize_kimi_parameters(p['model'],kwargs)
    assert actual['max_tokens']==65536 and actual['tool_choice']=='auto'
    if dialect=='qwen':
        assert actual['extra_body']==dict(enable_thinking=True,reasoning_effort='xhigh')
        assert 'reasoning_effort' not in actual
    elif dialect=='kimi':
        assert actual['reasoning_effort']=='high' and 'extra_body' not in actual
    else:
        assert actual['extra_body']==dict(thinking=dict(type='enabled')) and actual['reasoning_effort']=='max'

@pytest.mark.parametrize('dialect,effort',CASES)
def test_named_phase_is_narrowed_without_mutating_original(dialect,effort):
    kwargs=dict(tools=tools(),tool_choice=dict(type='function',function=dict(name='inspect')),extra_body=dict(other='preserved'))
    before=copy.deepcopy(kwargs);actual=apply_request_policy(profile(dialect,effort),kwargs)
    assert kwargs==before
    assert actual['tool_choice']=='auto' and actual['tools']==[kwargs['tools'][0]]
    with pytest.raises(ValueError,match='not exposed'):
        validate_exposed_call('submit',dict(candidate_id='a'),actual['tools'])
    with pytest.raises(ValueError,match='whitelist'):
        validate_exposed_call('inspect',dict(candidate_id='other'),actual['tools'])


def test_editor_auto_preserves_the_forced_tool_and_candidate_gate():
    choice,exposed=require_native_call(dict(function=dict(name='inspect')),tools(),tool_choice_policy='auto')
    assert choice=='auto' and exposed==[tools()[0]]
    assert require_native_call('required',tools(),tool_choice_policy='auto')[0]=='auto'

@pytest.mark.parametrize('dialect,effort',[('thinking','medium'),('kimi','medium'),('qwen','max')])
def test_wrong_model_effort_rejected_before_network(dialect,effort):
    with pytest.raises(ValueError,match='reasoning_effort'):validate_request_policy(profile(dialect,effort))


def test_no_silent_thinking_budget_or_disable_conflict():
    p=profile('qwen','xhigh');p['thinking_budget']=4096
    with pytest.raises(ValueError,match='thinking_budget'):validate_request_policy(p)
    p=profile('thinking','max');p['required_tool_thinking']='disabled'
    with pytest.raises(ValueError,match='conflicts'):validate_request_policy(p)
    with pytest.raises(ValueError,match='thinking_budget'):
        apply_request_policy(profile('qwen','xhigh'),dict(extra_body=dict(thinking_budget=4096)))


def test_profile_switch_clears_all_managed_fields(monkeypatch):
    p=dict(api_key='fake-only',base_url='https://example.invalid',model='unmanaged',timeout=240,max_tokens=8192)
    monkeypatch.setattr(service,'get_profile',lambda *a,**kw:p)
    keys=['ADVERSARIAL_REQUEST_DIALECT','ADVERSARIAL_THINKING_MODE','ADVERSARIAL_TOOL_CHOICE_POLICY','ADVERSARIAL_REASONING_EFFORT']
    env=service.backend_api_env(['python',str(CORE/'editing_backend.py')],dict.fromkeys(keys,'stale'))
    assert not any(k in env for k in keys)


def test_main_preserves_opaque_provider_field_for_correction():
    first=dict(message=dict(role='assistant',content='continue',reasoning_content='opaque-fixture-continuation',tool_calls=[]),finish_reason='stop')
    final=dict(message=dict(role='assistant',content=None,tool_calls=[dict(id='done',type='function',function=dict(name='finish',arguments=json.dumps(dict(reason='complete'))))]),finish_reason='tool_calls')
    client=Mock();client.call_functions.side_effect=[first,final]
    assert Orchestrator(client).decide('fixture',{})['action']=='finish'
    messages=client.call_functions.call_args.args[0]
    assert any(m.get('reasoning_content')=='opaque-fixture-continuation' for m in messages)
    assert all('opaque-fixture-continuation' not in m.get('content','') for m in messages if isinstance(m.get('content'),str))

@pytest.mark.parametrize('dialect,effort',CASES)
def test_cli_saves_managed_policy_without_credentials(tmp_path,dialect,effort):
    from scripts.api_profile import main
    path=tmp_path/'api.json';path.write_text('{}')
    assert main(['--config',str(path),'add','fixture','--base-url','https://example.invalid','--model','fixture-model','--key-env','FAKE_TEST_KEY','--request-dialect',dialect,'--thinking-mode','enabled','--reasoning-effort',effort,'--tool-choice-policy','auto'])==0
    saved=json.loads(path.read_text())['fixture'];api_config.validate_llm_profile(saved)
    assert saved['reasoning_effort']==effort and 'api_key' not in saved


@pytest.mark.parametrize('dialect,native', [
    ('thinking', ('low','high','max')), ('kimi', ('low','high','max')),
    ('qwen', ('low','medium','xhigh')),
])
def test_project_levels_map_to_supported_native_effort(monkeypatch,dialect,native):
    from mcp_servers.scene_editing.core.adversarial.llm_request_policy import effective_reasoning_effort
    p=profile(dialect,native[1])
    for level,expected in zip(('low','medium','high'),native):
        monkeypatch.setenv('LLM_REASONING_EFFORT',level)
        actual=apply_request_policy(p,dict(extra_body=dict(thinking=dict(type='disabled'))))
        effort=actual['extra_body']['reasoning_effort'] if dialect=='qwen' else actual['reasoning_effort']
        assert effort==expected
        assert effective_reasoning_effort(p)==expected
    validate_request_policy(p)


def test_unsupported_managed_dialect_fails_before_request():
    with pytest.raises(ValueError,match='request_dialect'):
        validate_request_policy(profile('openai','xhigh'))
