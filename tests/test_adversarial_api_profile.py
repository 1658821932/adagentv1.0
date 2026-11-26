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
        return dict(api_key='test-project-secret',base_url='https://example.invalid/v1',model='test-model',timeout=600,max_tokens=8192)
    monkeypatch.setattr(service,'get_profile',get_profile)
    env={'ADVERSARIAL_API_KEY':'stale-key','CARLA_GS_LLM_CONFIG':'/unused/old-profile'}
    injected=service.backend_api_env(['python',str(CORE/'editing_backend.py')],env)
    from common.api_profiles import profile_argument
    assert calls==[(profile_argument('attack_editor'),True)]
    assert injected['ADVERSARIAL_API_KEY']=='test-project-secret'
    assert injected['ADVERSARIAL_API_MODEL']=='test-model'
    assert env['ADVERSARIAL_API_KEY']=='stale-key'
    for key,value in injected.items():monkeypatch.setenv(key,value)
    assert loader()()==dict(api_key='test-project-secret',base_url='https://example.invalid/v1',model='test-model',timeout=600,max_tokens=8192)


def test_external_backend_does_not_require_project_credentials(monkeypatch):
    def fail(*args,**kwargs):raise AssertionError('must not load bundled profile')
    monkeypatch.setattr(service,'get_profile',fail)
    assert service.backend_api_env(['custom-backend'],{'VALUE':'test'})=={'VALUE':'test'}


def test_attack_budget_identity_survives_different_job_directories(tmp_path,monkeypatch):
    import json
    command=[str(CORE/'editing_backend.py')]
    monkeypatch.setattr(service,'get_profile',lambda *a,**k:dict(api_key='offline',base_url='https://example.invalid',model='offline',timeout=300,max_tokens=128))
    monkeypatch.setattr(service,'public_pipeline_root',lambda:None)
    paths=[]
    for name in ('first','second'):
        job=tmp_path/'jobs'/name;job.mkdir(parents=True)
        (job/'manifest.json').write_text(json.dumps(dict(scope='run_fixture')))
        monkeypatch.setenv('ADAGENT_JOB_DIR',str(job))
        paths.append(service.backend_api_env(command,{})['ADVERSARIAL_ATTACK_SESSION'])
    assert paths[0]==paths[1]==str(tmp_path/'jobs/attack-budgets/run_fixture')
    (job/'manifest.json').write_text(json.dumps(dict(scope='../outside')))
    with pytest.raises(ValueError,match='execution scope'):service.backend_api_env(command,{})


@pytest.mark.parametrize('choice',[None,'auto','required'])
def test_image_transport_capability_is_injected_and_stale_values_are_cleared(monkeypatch,choice):
    profile=dict(api_key='fake-test-key',base_url='https://example.invalid/v1',model='test',timeout=240,max_tokens=8192)
    if choice is not None:profile['image_tool_choice']=choice
    monkeypatch.setattr(service,'get_profile',lambda *a,**kw:profile)
    env=service.backend_api_env(['python',str(CORE/'editing_backend.py')],{'ADVERSARIAL_IMAGE_TOOL_CHOICE':'auto'})
    assert env.get('ADVERSARIAL_IMAGE_TOOL_CHOICE')==choice
    for key in ['ADVERSARIAL_API_KEY','ADVERSARIAL_API_BASE_URL','ADVERSARIAL_API_MODEL']:
        monkeypatch.setenv(key,env[key])
    if choice is None:monkeypatch.delenv('ADVERSARIAL_IMAGE_TOOL_CHOICE',raising=False)
    else:monkeypatch.setenv('ADVERSARIAL_IMAGE_TOOL_CHOICE',choice)
    assert loader()().get('image_tool_choice')==choice


def test_core_does_not_fall_back_to_old_profile(monkeypatch):
    for key in ['ADVERSARIAL_API_KEY','ADVERSARIAL_API_BASE_URL','ADVERSARIAL_API_MODEL']:
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv('CARLA_GS_LLM_CONFIG','/unused/old-profile')
    with pytest.raises(RuntimeError,match='host adapter'):
        loader()()


def test_public_output_path_is_owned_by_adapter(monkeypatch, tmp_path):
    monkeypatch.setattr(service, 'get_profile', lambda *a, **k: dict(
        api_key='test',base_url='https://example.invalid/v1',model='test',timeout=600,max_tokens=8192))
    root = tmp_path / 'public'
    monkeypatch.setattr(service, 'public_pipeline_root', lambda: root)
    env = {'ADVERSARIAL_FULL_PIPELINE_ROOT': '/stale'}
    child = service.backend_api_env(['python',str(CORE/'editing_backend.py')],env)
    assert child['ADVERSARIAL_FULL_PIPELINE_ROOT'] == str(root)
    assert env['ADVERSARIAL_FULL_PIPELINE_ROOT'] == '/stale'
    monkeypatch.setattr(service, 'public_pipeline_root', lambda: None)
    child = service.backend_api_env(['python',str(CORE/'editing_backend.py')],env)
    assert 'ADVERSARIAL_FULL_PIPELINE_ROOT' not in child
    assert service.backend_api_env(['custom-backend'],env) == env


def test_public_output_relative_path_uses_project_root(monkeypatch, tmp_path):
    from mcp_servers.scene_editing.adapter import config
    monkeypatch.setattr(config, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(config, 'settings', lambda *a: {'public_pipeline_root':'outputs/rendered'})
    assert config.public_pipeline_root() == tmp_path / 'outputs/rendered'
    monkeypatch.setattr(config, 'settings', lambda *a: {})
    assert config.public_pipeline_root() is None
