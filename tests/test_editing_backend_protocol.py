"""The backend stdout channel contains a single JSON response, including on failure."""
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / 'mcp_servers/scene_editing/core/adversarial/editing_backend.py'


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.syspath_prepend(str(BACKEND.parent))
    exports = {
        'experiment_runtime': ['atomic_json'],
        'review_attack_workflow': ['review'],
        'run_adversarial_loop': ['pipeline_video_dir', 'run_command', 'run_constructor', 'run_independent', 'run_render'],
        'run_attack_workflow': ['run'], 'run_ego_conflict_brief': ['run'],
        'run_ego_reference': ['ego_path_from_frames'], 'run_ego_visual_editor_agent': ['run'],
    }
    for name, names in exports.items():
        module = types.ModuleType(name)
        for attribute in names:
            setattr(module, attribute, lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location('protocol_backend', BACKEND)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('stage,status', [('attack_editing', 'completed'), ('ego_editing', 'completed'),
                                         ('attack_editing', 'attack_not_submitted')])
def test_diagnostics_do_not_pollute_json(backend, monkeypatch, capsys, tmp_path, stage, status):
    def execute(arguments, output):
        print('internal progress before result')
        print('another diagnostic')
        return {'status': status, 'detail': '结果'}
    monkeypatch.setattr(backend, 'attack_output', execute)
    monkeypatch.setattr(backend, 'ego_output', execute)
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(dict(stage=stage, arguments={}, output_dir=str(tmp_path)))))
    backend.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {'status': status, 'detail': '结果'}
    assert 'internal progress' in captured.err
    assert len(captured.out.splitlines()) == 1


def test_exception_after_logging_is_json(backend, monkeypatch, capsys, tmp_path):
    def fail(*args):
        print('starting stage')
        raise ValueError('test stage failure')
    monkeypatch.setattr(backend, 'attack_output', fail)
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(dict(stage='attack_editing', arguments={}, output_dir=str(tmp_path)))))
    backend.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == dict(status='backend_error', error_type='ValueError', error='test stage failure')
    assert 'starting stage' in captured.err


def test_configuration_error_exports_setting_values_and_recovery(backend,monkeypatch,capsys,tmp_path):
    from attack_configuration import AttackConfigurationError
    def fail(*args):
        raise AttackConfigurationError('Conflicting task and acceptance setting: observation_end_s',
            setting='observation_end_s',task_value=0.,acceptance_value=1.5)
    monkeypatch.setattr(backend,'attack_output',fail)
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps(dict(stage='attack_editing',arguments={},output_dir=str(tmp_path)))))
    backend.main();result=json.loads(capsys.readouterr().out)
    assert result['error_code']=='editing_configuration_error'
    assert result['failure_stage']=='configuration' and result['retryable'] is False
    assert result['error_details']['task_value']==0.
    assert result['error_details']['acceptance_value']==1.5
    assert 'changing the scene' in result['recovery_hint']


@pytest.mark.parametrize("prebuilt", [False, True])
@pytest.mark.parametrize("operation", ["add_vehicle", "edit_existing"])
def test_packaged_commands_exist(monkeypatch, tmp_path, prebuilt, operation):
    import ast
    source = BACKEND.with_name('run_adversarial_loop.py')
    tree = ast.parse(source.read_text())
    names = {'run_constructor', 'run_independent', 'run_render'}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    calls = []
    def capture(command, *args):
        calls.append(command)
        assert Path(command[1]).is_file(), command[1]
    def write(path, data):
        Path(path).write_text(json.dumps(data))
    namespace = dict(__file__=str(source), Path=Path, sys=sys, os=__import__('os'), json=json,
                     atomic_json=write, run_command=capture)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    namespace['run_constructor'](tmp_path, 'target', tmp_path/'constructor', {}, 24, 300)
    review = tmp_path/'review.json';write(review, {'status':'passed'})
    namespace['run_independent'](tmp_path, tmp_path, 'edit_001', 'target', review, 300)
    write(tmp_path/'render-report.json', {'status':'passed'})
    write(tmp_path/'manual-explained-validation.json', {'status':'passed'})
    render_settings=dict(enabled=True, python=sys.executable, worldengine_dir=str(tmp_path), system_lib=str(tmp_path))
    if prebuilt:
        render_settings['gsplat_extension']=str(tmp_path/'gsplat_cuda.so')
    namespace['run_render']({'render':render_settings, 'scene':str(tmp_path/'original')},
        dict(target=dict(operation=operation,track_id='target'),frozen_scene_path=str(tmp_path),
             selected_candidate_id='attack_001'), tmp_path, tmp_path,'edit_001',tmp_path)
    if operation == 'edit_existing':
        render_command = calls[2]
        assert render_command[render_command.index('--original-scene')+1] == str(tmp_path/'original')
        assert render_command[render_command.index('--attack-candidate')+1] == 'attack_001'
    assert len(calls)==4
    if prebuilt:
        assert Path(calls[2][1]).name=="render_runtime.py"
        assert calls[2][2:4]==["--extension",str(tmp_path/"gsplat_cuda.so")]
        assert Path(calls[2][4]).name=="render_ego_mtgs_three_view.py"


def test_pdm_runner_owns_output_creation(monkeypatch, tmp_path):
    spec=importlib.util.spec_from_file_location('protocol_pdm', BACKEND.with_name('pdm_tool.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def execute(command, **kwargs):
        output=Path(command[command.index('--output')+1])
        assert not output.exists()
        output.mkdir();(output/'attack').mkdir()
        clear=dict(contact_samples=0,road_violation_samples=0)
        (output/'summary.json').write_text(json.dumps(dict(results=dict(baseline=clear,attack=clear))))
        (output/'attack/observations.json').write_text(json.dumps([dict(target_in_pdm=True,observation_timestamp_us=1,max_history_time_us=1)]))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',execute)
    args=dict(challenge_baseline_scene_path=str(tmp_path),challenge_attack_scene_path=str(tmp_path),
              challenge_history_s=3,challenge_target_track='target',challenge_target_is_new=True)
    assert module.run_pdm_challenge(args,tmp_path/'output',tmp_path/'pdm.log')['pdm_defense_succeeded']


def test_materialization_respects_configured_task_root(tmp_path):
    import ast
    source = BACKEND.with_name('add_vehicle_scene.py')
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'add_vehicle')
    node.decorator_list = []
    namespace = dict(Path=Path, experiments_root=lambda: tmp_path/'task-output',
                     policy=types.SimpleNamespace(OFFICIAL_MARGIN_M=.1), json=json)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    function = namespace['add_vehicle']
    with pytest.raises(ValueError, match='new directory'):
        function(tmp_path/'source', tmp_path/'outside', 0, 'target', [4,2,1])
    # An allowed task directory advances to reading source frames, rather than being
    # rejected merely because it is outside the former package experiments directory.
    with pytest.raises(FileNotFoundError):
        function(tmp_path/'source', tmp_path/'task-output'/'materialized', 0, 'target', [4,2,1])
    existing=tmp_path/'task-output'/'existing';existing.mkdir(parents=True)
    with pytest.raises(ValueError, match='new directory'):
        function(tmp_path/'source', existing, 0, 'target', [4,2,1])


def test_stage_command_creates_parent_and_preserves_existing_log(tmp_path):
    import ast
    import subprocess
    source=BACKEND.with_name('run_adversarial_loop.py')
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run_command')
    namespace=dict(Path=Path,subprocess=subprocess,ROOT=tmp_path)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    log=tmp_path/'nested'/'run.log'
    namespace['run_command']([sys.executable,'-c','print("stage complete")'],log,10)
    assert 'stage complete' in log.read_text()
    with pytest.raises(FileExistsError):namespace['run_command']([sys.executable,'-c','pass'],log,10)


def test_revision_feedback_reaches_attack_without_changing_contract(backend, monkeypatch, tmp_path):
    feedback=dict(stage='pdm',outcome='pdm_defense_succeeded',evidence=dict(min_target_gap_m=.425))
    contract=dict(schema_version='attack_task.v2',objective=dict(keyframes=[]))
    captured={}
    from attack_configuration import policy as actual_policy
    monkeypatch.setattr(backend.policy,'for_new_run',actual_policy.for_new_run,raising=False)
    def attack(scene,folder,instruction,assets,constraints,**kwargs):
        captured.update(constraints)
        archive=folder/'trajectory_editing/output.json'
        archive.parent.mkdir(parents=True)
        archive.write_text(json.dumps(dict(status='not_submitted',stop_reason='finished_without_submission')))
        return dict(status='not_submitted')
    monkeypatch.setattr(backend,'run_attack',attack)
    arguments=dict(scene_ref=dict(sample_root=str(tmp_path)),instruction='Keep the vehicle stationary',
        operation='add_vehicle',maneuver='stationary',task_contract=contract,
        revision_feedback=feedback,timeout=3600,vadv2_self_test=False)
    result=backend.attack_output(arguments,tmp_path)
    assert result['status']=='unable_to_complete'
    assert captured['revision_feedback']==feedback
    assert captured['task_contract']==contract
    assert captured['stationary_target'] is True
    assert captured['observation_end_s']==1.5
    assert captured['attack_prefix_lock_s']==0.


def test_auto_options_reach_backend_without_a_fixed_operation_or_stationary_contract(backend,monkeypatch,tmp_path):
    from attack_configuration import policy as actual_policy
    monkeypatch.setattr(backend.policy,'for_new_run',actual_policy.for_new_run,raising=False)
    captured={}
    def attack(scene,folder,instruction,assets,constraints,**kwargs):
        captured.update(constraints=constraints,kwargs=kwargs,assets=assets,instruction=instruction)
        archive=folder/'trajectory_editing/output.json';archive.parent.mkdir(parents=True)
        archive.write_text(json.dumps(dict(status='not_submitted',stop_reason='finished_without_submission')))
        return dict(status='not_submitted')
    monkeypatch.setattr(backend,'run_attack',attack)
    options=[dict(option_id='new',operation='add_vehicle',maneuver='stationary',instruction='A possible stationary lead.')]
    vehicle=dict(asset_path=str(tmp_path/'vehicle.ply'),size_m=[4.2,1.8,1.6])
    arguments=dict(scene_ref=dict(sample_root=str(tmp_path)),instruction='Create a braking conflict.',
        operation='auto',plan_options=options,vehicle=vehicle,timeout=3600,vadv2_self_test=False)
    result=backend.attack_output(arguments,tmp_path)
    assert result['status']=='unable_to_complete'
    assert captured['constraints']['plan_options']==options
    assert not captured['constraints'].get('stationary_target')
    assert captured['kwargs']['required_operation'] is None
    assert captured['kwargs']['required_track'] is None
    assert captured['assets'][0]['path']==vehicle['asset_path']
    assert captured['instruction']==arguments['instruction']


def test_new_public_attack_rejects_removed_class_before_any_search(backend,monkeypatch,tmp_path):
    observed=[]
    monkeypatch.setattr(backend,'run_attack',lambda *a,**kw:observed.append(True))
    with pytest.raises(ValueError,match='categories are removed'):
        backend.attack_output(dict(scene_ref={'sample_root':str(tmp_path)},timeout=3600,operation='edit_existing',
            task_contract={'objective':{'interaction':'either'}}),tmp_path)
    assert observed==[]
