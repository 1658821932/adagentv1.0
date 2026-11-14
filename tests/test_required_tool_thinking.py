"""Explicit forced-tool thinking compatibility never weakens native restrictions."""
import copy
import importlib.util
from pathlib import Path

import pytest
import api_config
from mcp_servers.scene_editing.adapter import service
from mcp_servers.scene_editing.core.adversarial.kimi_request_parameters import normalize_kimi_parameters

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


@pytest.mark.parametrize('choice',['required',{'type':'function','function':{'name':'select_vehicle'}}])
def test_explicit_capability_disables_thinking_without_changing_tools(monkeypatch,choice):
    monkeypatch.setenv('ADVERSARIAL_REQUIRED_TOOL_THINKING','disabled')
    monkeypatch.delenv('ADVERSARIAL_API_MAX_TOKENS',raising=False)
    request=dict(tool_choice=choice,tools=[dict(type='function',function=dict(name='select_vehicle'))],extra_body=dict(thinking=dict(type='enabled'),other='kept'),reasoning_effort='low')
    snapshot=copy.deepcopy(request)
    actual=normalize_kimi_parameters('deepseek-flash',request)
    assert actual['tool_choice']==choice and actual['tools']==request['tools']
    assert actual['extra_body']==dict(thinking=dict(type='disabled'),other='kept')
    assert request==snapshot


@pytest.mark.parametrize('choice',['auto','none',None])
def test_unforced_requests_preserve_thinking(monkeypatch,choice):
    monkeypatch.setenv('ADVERSARIAL_REQUIRED_TOOL_THINKING','disabled')
    request=dict(tool_choice=choice,extra_body=dict(thinking=dict(type='enabled')))
    assert normalize_kimi_parameters('deepseek-flash',request)['extra_body']==request['extra_body']


def test_other_profiles_retain_existing_behavior(monkeypatch):
    monkeypatch.delenv('ADVERSARIAL_REQUIRED_TOOL_THINKING',raising=False)
    request=dict(tool_choice='required',extra_body=dict(thinking=dict(type='enabled')))
    assert normalize_kimi_parameters('deepseek-v4.1-flash-1017',request)['extra_body']==request['extra_body']


@pytest.mark.parametrize('mode',[None,'preserve','disabled'])
def test_profile_injection_clears_stale_capability(monkeypatch,mode):
    profile=dict(api_key='fake-key-only',base_url='https://example.invalid',model='model',timeout=240,max_tokens=8192)
    if mode is not None:profile['required_tool_thinking']=mode
    monkeypatch.setattr(service,'get_profile',lambda *a,**kw:profile)
    env=service.backend_api_env(['python',str(CORE/'editing_backend.py')],{'ADVERSARIAL_REQUIRED_TOOL_THINKING':'disabled'})
    assert env.get('ADVERSARIAL_REQUIRED_TOOL_THINKING')==mode
    for key in ['ADVERSARIAL_API_KEY','ADVERSARIAL_API_BASE_URL','ADVERSARIAL_API_MODEL']:
        monkeypatch.setenv(key,env[key])
    if mode is None:monkeypatch.delenv('ADVERSARIAL_REQUIRED_TOOL_THINKING',raising=False)
    else:monkeypatch.setenv('ADVERSARIAL_REQUIRED_TOOL_THINKING',mode)
    spec=importlib.util.spec_from_file_location('forced_thinking_loader',CORE/'llm_profile.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.load_llm_profile().get('required_tool_thinking')==mode


@pytest.mark.parametrize('mode',['auto','enabled',False,1])
def test_invalid_capabilities_fail_before_network(mode):
    profile=dict(base_url='https://example.invalid',model='model',api_key_env='FAKE_KEY',timeout=240,max_tokens=8192,required_tool_thinking=mode)
    with pytest.raises(ValueError,match='required_tool_thinking'):api_config.validate_llm_profile(profile)


def test_kimi_named_tool_restriction_still_normalized(monkeypatch):
    monkeypatch.delenv('ADVERSARIAL_REQUIRED_TOOL_THINKING',raising=False)
    tools=[dict(type='function',function=dict(name=n)) for n in ['a','b']]
    actual=normalize_kimi_parameters('kimi-k3',dict(tool_choice=dict(type='function',function=dict(name='b')),tools=tools,extra_body=dict(thinking=dict(type='enabled'))))
    assert actual['tool_choice']=='required' and actual['tools']==[tools[1]]
    assert 'extra_body' not in actual
