"""Tool06 accepts configured defaults through native validation and launch."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from agent.orchestrator import Orchestrator
from tools.tool06_policy_runner.schemas import Input


def configure(tmp_path, monkeypatch):
    from mcp_servers.worldengine.adapter import config
    cfg=tmp_path/'policy.py';cfg.write_text('# offline fixture')
    weights=tmp_path/'weights.pth';weights.write_bytes(b'offline fixture, never loaded')
    monkeypatch.setattr(config,'DEFAULT_POLICY_CONFIG',cfg)
    monkeypatch.setattr(config,'DEFAULT_POLICY_CHECKPOINT',weights)
    return config,cfg,weights


@pytest.mark.parametrize('null_fields',[False,True])
def test_native_input_without_policy_identity_uses_configured_defaults(tmp_path,monkeypatch,null_fields):
    config,cfg,weights=configure(tmp_path,monkeypatch)
    args=dict(execution_mode='native',num_history=4,max_frame_per_scene=16,with_dense_reward_manager=True,
              scenario_data_folder_path='/fixture/expert',asset_folder_path='/fixture/assets')
    if null_fields:
        args.update(config_path=None,checkpoint_path=None,model_name=None,data_type=None)
    validated=Input.model_validate(args)
    actual=config.resolve_policy_inputs(validated.model_dump())
    assert actual['config_path']==str(cfg) and actual['checkpoint_path']==str(weights)
    assert actual['model_name']==config.DEFAULT_MODEL_NAME and actual['data_type']==config.DEFAULT_DATA_TYPE
    assert actual['max_frame_per_scene']==16 and actual['with_dense_reward_manager'] is True
    assert not args.get('config_path')


def test_explicit_overrides_and_missing_artifact_errors_are_preserved(tmp_path,monkeypatch):
    config,cfg,weights=configure(tmp_path,monkeypatch)
    args=dict(config_path=str(cfg),checkpoint_path=str(weights),model_name='requested',data_type='requested-data')
    assert config.resolve_policy_inputs(args)==args
    missing=tmp_path/'missing.pth'
    with pytest.raises(FileNotFoundError,match='Policy model artifact is missing') as error:
        config.resolve_policy_inputs(dict(args,checkpoint_path=str(missing)))
    assert str(missing) in str(error.value)
    weights.unlink()
    with pytest.raises(FileNotFoundError):config.resolve_policy_inputs({})


def test_real_orchestrator_accepts_the_failed_native_call_without_a_correction():
    args=dict(execution_mode='native',scenario_data_folder_path='/fixture/native_expert',
        asset_folder_path='/fixture/native_expert/assets',scenario_pkl_file_name='all_scenarios.pkl',
        with_dense_reward_manager=True,num_history=4,max_frame_per_scene=16,
        collision_target_track='synthetic_vehicle_001',react_type='NR',added_name='attack_expert')
    response=dict(message=dict(tool_calls=[dict(id='fixture',type='function',function=dict(name='policy_runner',
        arguments=json.dumps(dict(arguments=args,reason='Use expert package with configured policy defaults'))))]),
        finish_reason='tool_calls')
    calls=[]
    client=SimpleNamespace(call_functions=lambda *parameters:(calls.append(parameters) or response))
    decision=Orchestrator(client,decision_correction_attempts=0).decide('Continue 07→09',{})
    assert decision['action']=='call_tool' and decision['tool_name']=='policy_runner'
    assert decision['arguments']==args and len(calls)==1


@pytest.mark.parametrize('dense',[False,True])
def test_registry_reaches_native_runner_with_defaults_and_collects_dense_data(tmp_path,monkeypatch,dense):
    from mcp_servers.worldengine.adapter import client
    from tools.registry import call_tool
    config,cfg,weights=configure(tmp_path,monkeypatch)
    monkeypatch.setenv('ADAGENT_RUNTIME_ROOT',str(tmp_path/'runtime'))
    monkeypatch.setenv('WORLDENGINE_LIBRARY_DIR',str(tmp_path))
    monkeypatch.setenv('LD_LIBRARY_PATH','/fixture/existing-library')
    result_dir=tmp_path/'result'
    rollout=result_dir/'WE_output/openscene_format';rollout.mkdir(parents=True)
    calls=[]
    if dense:
        (rollout/'pdms_pkl').mkdir()
        (rollout/'pdms_pkl/fixture_scores.pkl').write_bytes(b'offline fixture')
        (result_dir/'plan_traj').mkdir()
        (result_dir/'plan_traj/plan_idx.csv').write_text('frame_idx,plan_idx\n0,0\n')
    def runner(command,**kwargs):
        calls.append((command,kwargs))
        assert kwargs['env']['LD_LIBRARY_PATH']==str(tmp_path)+':/fixture/existing-library'
        if '--request' in command:
            request=json.loads(Path(command[command.index('--request')+1]).read_text())
            assert request['config_path']==str(cfg) and request['checkpoint_path']==str(weights)
            return subprocess.CompletedProcess(command,0,json.dumps(dict(
                rlft_training_data=None,rlft_data_report=dict(status='no_eligible_scenes'))),'')
        args=json.loads(kwargs['input'])
        assert args['config_path']==str(cfg) and args['checkpoint_path']==str(weights)
        assert args['model_name']==config.DEFAULT_MODEL_NAME and args['data_type']==config.DEFAULT_DATA_TYPE
        assert args['max_frame_per_scene']==16 and args['with_dense_reward_manager']==dense
        return subprocess.CompletedProcess(command,0,json.dumps(dict(status='success',result=dict(
            result_dir=str(result_dir),reports=[dict(succeeded=True)]))),'')
    monkeypatch.setattr(client,'run_logged',runner)
    output=call_tool('policy_runner',dict(execution_mode='native',max_frame_per_scene=16,
        with_dense_reward_manager=dense,scenario_data_folder_path='/fixture/native_expert',
        asset_folder_path='/fixture/assets'),mode='real')
    assert output['status']=='success' and output['result']['policy_result']=='completed'
    assert len(calls)==(2 if dense else 1)
    if dense:
        assert (rollout/'pdms_pkl/plan_idx.csv').is_file()
        assert output['result']['rlft_data_report']['status']=='no_eligible_scenes'


def test_other_native_and_pdm_validation_remains_strict():
    with pytest.raises(ValueError,match='max_frame_per_scene must exceed'):
        Input.model_validate(dict(num_history=16,max_frame_per_scene=16))
    with pytest.raises(ValueError,match='stale override'):
        Input.model_validate(dict(max_frame_per_scene=24,num_history=4,sampling_poses=8))
    with pytest.raises(ValueError,match='baseline/attack scene paths'):
        Input.model_validate(dict(execution_mode='pdm_closed_loop'))
