"""Bundled editing agents use the host's profile without credential files."""
import importlib.util
from pathlib import Path

import pytest
from mcp_servers.scene_editing.adapter import service

ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'mcp_servers/scene_editing/core/adversarial'


def loader():
    spec=importlib.util.spec_from_file_location('test_profile_loader',CORE/'llm_profile.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.load_llm_profile


def test_adapter_injects_project_profile(monkeypatch):
    calls=[]
    def get_profile(name, *, require_key):
        calls.append((name,require_key))
        return dict(api_key='test-project-secret',base_url='https://example.invalid/v1',model='test-model')
    monkeypatch.setattr(service,'get_profile',get_profile)
    env={'ADVERSARIAL_API_KEY':'stale-key','CARLA_GS_LLM_CONFIG':'/unused/old-profile'}
    injected=service.backend_api_env(['python',str(CORE/'editing_backend.py')],env)
    assert calls==[('kimi',True)]
    assert injected['ADVERSARIAL_API_KEY']=='test-project-secret'
    assert injected['ADVERSARIAL_API_MODEL']=='test-model'
    assert env['ADVERSARIAL_API_KEY']=='stale-key'
    for key,value in injected.items():monkeypatch.setenv(key,value)
    assert loader()()==dict(api_key='test-project-secret',base_url='https://example.invalid/v1',model='test-model')


def test_external_backend_does_not_require_project_credentials(monkeypatch):
    def fail(*args,**kwargs):raise AssertionError('must not load bundled profile')
    monkeypatch.setattr(service,'get_profile',fail)
    assert service.backend_api_env(['custom-backend'],{'VALUE':'test'})=={'VALUE':'test'}


def test_core_does_not_fall_back_to_old_profile(monkeypatch):
    for key in ['ADVERSARIAL_API_KEY','ADVERSARIAL_API_BASE_URL','ADVERSARIAL_API_MODEL']:
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('CARLA_GS_LLM_CONFIG','/unused/old-profile')
    with pytest.raises(RuntimeError,match='host adapter'):
        loader()()
