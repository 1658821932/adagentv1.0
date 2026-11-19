"""Attack lock, observation history and event window are independent settings."""
import ast
import json
import sys
from pathlib import Path

import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
if str(CORE) not in sys.path:sys.path.insert(0,str(CORE))
import acceptance_policy as policy
from attack_configuration import AttackConfigurationError, normalize_constraints


def test_default_lock_history_and_event_window_remain_distinct():
    effective=policy.for_new_run()
    supplied=dict(observation_end_s=effective['observation_end_s'],attack_prefix_lock_s=0.,
        event_window_s=[effective['attack_window_start_s'],effective['attack_window_end_s']],acceptance_policy=effective)
    result=normalize_constraints(supplied)
    assert result['observation_end_s']==1.5
    assert result['attack_prefix_lock_s']==0.
    assert result['event_window_s']==[1.5,7.5]
    assert result['acceptance_policy']['observation_end_s']==1.5
    assert result==normalize_constraints(result)
    assert supplied['acceptance_policy']==effective


@pytest.mark.parametrize('setting,value', [('observation_end_s',0.),('attack_prefix_lock_s',.5),('official_margin_m',.2)])
def test_conflicts_in_the_same_setting_remain_explicit(setting,value):
    with pytest.raises(AttackConfigurationError) as exc:
        normalize_constraints({setting:value,'acceptance_policy':policy.for_new_run()})
    assert exc.value.code=='editing_configuration_error'
    assert exc.value.details['setting']==setting
    assert exc.value.details['task_value']==value
    assert 'acceptance_value' in exc.value.details


def test_zero_history_without_policy_is_not_a_way_to_unlock_attack_motion():
    with pytest.raises(AttackConfigurationError,match='use attack_prefix_lock_s=0'):
        normalize_constraints({'observation_end_s':0.})


def test_invalid_acceptance_settings_are_configuration_errors():
    with pytest.raises(AttackConfigurationError) as exc:
        normalize_constraints({'acceptance_policy':{'observation_end_s':0.}})
    assert exc.value.details['setting']=='acceptance_policy'


def test_archived_v1_boundary_keeps_its_explicit_history_and_lock():
    result=normalize_constraints({'acceptance_policy':{'policy_version':'reference_acceptance.v1','observation_end_s':3.}})
    assert result['observation_end_s']==result['attack_prefix_lock_s']==3.
    assert result==normalize_constraints(result)


@pytest.mark.parametrize('operation',['edit_existing','add_vehicle'])
def test_real_backend_to_workflow_reaches_request_preparation_without_an_api(tmp_path,monkeypatch,operation):
    import run_attack_workflow as workflow
    scene=tmp_path/'scene';(scene/'metadata').mkdir(parents=True)
    frames=[]
    for i in range(24):
        frames.append(dict(timestamp=1_000_000+i*500_000,token=str(i),
            ego2global=[[1.,0.,0.,i],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]))
    (scene/'metadata/frames.json').write_text(json.dumps(frames))
    output=tmp_path/'output';output.mkdir()
    monkeypatch.setenv('ADVERSARIAL_EXPERIMENTS_ROOT',str(output))
    class ReachedRequestPreparation(Exception):pass
    def profile():raise ReachedRequestPreparation
    monkeypatch.setattr(workflow,'load_llm_profile',profile)
    fn=next(n for n in ast.parse((CORE/'editing_backend.py').read_text()).body
        if isinstance(n,ast.FunctionDef) and n.name=='attack_output')
    namespace=dict(Path=Path,policy=policy,run_attack=workflow.run,
        SETTINGS={'defaults':{'attack_max_turns':6}})
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'backend.attack_output','exec'),namespace)
    args=dict(scene_ref={'sample_root':str(scene)},instruction='Fixture attack',operation=operation,
        constraints={},timeout=300,vadv2_self_test=False)
    if operation=='add_vehicle':
        asset=tmp_path/'fixture.ply';asset.write_text('offline asset, never rendered')
        args.update(vehicle={'asset_path':str(asset),'size_m':[4.,2.,1.5]},maneuver='stationary')
    with pytest.raises(ReachedRequestPreparation):namespace['attack_output'](args,output)
    request=json.loads((output/'attack/input.json').read_text())
    assert request['constraints']['observation_end_s']==1.5
    assert request['constraints']['attack_prefix_lock_s']==0.
    assert request['constraints']['event_window_s']==[1.5,7.5]
    assert request['required_operation']==operation
    assert not (output/'attack/object_selection').exists()
