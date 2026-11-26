"""Editing must not start an unrequested policy simulation."""
import ast
from pathlib import Path
from unittest.mock import Mock
import pytest

ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'mcp_servers/scene_editing/core/adversarial'


def test_attack_backend_rejects_internal_pdm_before_execution(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(CORE))
    import acceptance_policy
    tree=ast.parse((CORE/'editing_backend.py').read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='attack_output')
    run=Mock(side_effect=AssertionError('Editing must not launch PDM or an attack after invalid options'))
    namespace=dict(Path=Path,policy=acceptance_policy,run_attack=run)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'editing_backend.attack_output','exec'),namespace)
    for flag in ('pdm_self_check','pdm_defense_feedback','pdm_ignore_traffic_lights'):
        args=dict(scene_ref=dict(sample_root=str(tmp_path)),**{flag:True})
        with pytest.raises(ValueError,match='does not run PDM'):
            namespace['attack_output'](args,tmp_path/'output')
    run.assert_not_called()


def test_native_policy_tool_does_not_dispatch_the_legacy_challenge(monkeypatch):
    import tools.tool06_policy_runner.real as tool
    native=Mock(return_value=dict(result_dir='unit-test-native-result'))
    challenge=Mock(side_effect=AssertionError('Native mode must not run the independent PDM challenge'))
    monkeypatch.setattr(tool,'run_policy',native)
    monkeypatch.setattr(tool,'run_pdm',challenge)
    arguments=dict(execution_mode='native',config_path='test-config',checkpoint_path='test-checkpoint')
    result=tool.run(arguments)
    assert result['status']=='success'
    native.assert_called_once_with(arguments)
    challenge.assert_not_called()


def test_attack_public_input_has_no_internal_pdm_options():
    from mcp_servers.scene_editing.adapter.schemas import AttackInput
    assert not {'pdm_self_check','pdm_ignore_traffic_lights','pdm_probe_budget','pdm_defense_feedback'} & set(AttackInput.model_fields)
