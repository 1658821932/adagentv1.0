"""Probe gating, immutable candidate identity and budget inheritance without GPU/API."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'


@pytest.fixture
def probe_module(monkeypatch):
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    exports={
        'experiment_runtime':dict(atomic_json=write),
        'pdm_tool':dict(run_pdm_challenge=lambda *args:None),
        'run_ego_reference':dict(EGO_FRONT=4.,EGO_REAR=1.,EGO_WIDTH=2.,polygon=lambda *args:None),
        'workflow_timing':dict(timed_operation=lambda *args:lambda function:function),
    }
    for name,values in exports.items():
        module=types.ModuleType(name)
        for key,value in values.items():setattr(module,key,value)
        monkeypatch.setitem(sys.modules,name,module)
    spec=importlib.util.spec_from_file_location('test_probe_module',CORE/'attack_pdm_probe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def editor(tmp_path):
    node=dict(xy=np.zeros((2,2)),yaw=np.zeros(2),result=dict(reference_screen_passed=True,
        construction_goal=dict(goal_met=True)))
    return types.SimpleNamespace(scene=tmp_path/'scene',run_dir=tmp_path/'editor',
        constraints=dict(pdm_probe_root=str(tmp_path/'shared'),pdm_probe_budget=2),
        target=dict(operation='add_vehicle',track_id='target'),nodes={'seed_000':node})


def prepare(module,monkeypatch,manager,calls,safe=True):
    monkeypatch.setattr(manager,'_materialize',lambda candidate,folder:folder/'frozen_scene')
    monkeypatch.setattr(manager,'_feedback',lambda *args:dict(attack={},overlay_path='/overlay.png'))
    def run(arguments,output,log):
        calls.append(arguments)
        return dict(pdm_defense_succeeded=safe,summary=dict(baseline_reused=bool(arguments.get('challenge_baseline_cache_path'))))
    monkeypatch.setattr(module,'run_pdm_challenge',run)


def test_bad_geometry_spends_no_probe_budget(probe_module,tmp_path):
    e=editor(tmp_path);e.nodes['seed_000']['result']['reference_screen_passed']=False
    manager=probe_module.PDMProbe(e)
    with pytest.raises(ValueError,match='reference checks'):manager.probe('seed_000')
    assert manager.context()['calls']==0


def test_repeat_candidate_reuses_result_and_safe_candidate_cannot_submit(probe_module,monkeypatch,tmp_path):
    manager=probe_module.PDMProbe(editor(tmp_path));calls=[]
    prepare(probe_module,monkeypatch,manager,calls)
    assert manager.probe('seed_000')['status']=='completed'
    assert manager.probe('seed_000')['cached']
    assert len(calls)==1 and manager.context()['calls']==1
    with pytest.raises(ValueError,match='defended'):manager.require_challenge('seed_000')


def test_reselection_inherits_budget_and_baseline_cache(probe_module,monkeypatch,tmp_path):
    calls=[]
    first=probe_module.PDMProbe(editor(tmp_path));prepare(probe_module,monkeypatch,first,calls)
    first.probe('seed_000')
    second=probe_module.PDMProbe(editor(tmp_path));prepare(probe_module,monkeypatch,second,calls)
    assert second.probe('seed_000')['baseline_reused']
    third=probe_module.PDMProbe(editor(tmp_path))
    assert third.probe('seed_000')['status']=='budget_exhausted'
    assert len(calls)==2 and third.context()['remaining']==0


def test_passing_probe_rejects_mutated_candidate_or_target(probe_module,monkeypatch,tmp_path):
    e=editor(tmp_path);manager=probe_module.PDMProbe(e);prepare(probe_module,monkeypatch,manager,[],safe=False)
    with pytest.raises(ValueError,match='before submitting'):manager.require_challenge('seed_000')
    manager.probe('seed_000')
    assert manager.require_challenge('seed_000')['pdm_defense_succeeded'] is False
    e.nodes['seed_000']['xy'][0,0]=.01
    with pytest.raises(ValueError,match='different candidate version'):manager.require_challenge('seed_000')
    e.nodes['seed_000']['xy'][0,0]=0.
    e.target['track_id']='other'
    with pytest.raises(ValueError,match='different candidate version'):manager.require_challenge('seed_000')


def test_engine_failure_is_not_an_attack_outcome(probe_module,monkeypatch,tmp_path):
    manager=probe_module.PDMProbe(editor(tmp_path));prepare(probe_module,monkeypatch,manager,[])
    def fail(*args):raise RuntimeError('missing simulator dependency')
    monkeypatch.setattr(probe_module,'run_pdm_challenge',fail)
    with pytest.raises(RuntimeError,match='execution failed'):manager.probe('seed_000')
    assert manager.context()['calls']==1 and manager.results=={}


def test_previous_branch_safe_result_is_not_current_probe(probe_module,monkeypatch,tmp_path):
    first=probe_module.PDMProbe(editor(tmp_path));prepare(probe_module,monkeypatch,first,[])
    first.probe('seed_000')
    current=probe_module.PDMProbe(editor(tmp_path))
    assert current.context()['latest_feedback']['pdm_defense_succeeded'] is True
    assert current.context()['untested_eligible_candidate_ids']==['seed_000']
    with pytest.raises(ValueError,match='prior-run/branch'):
        current.require_reselection_evidence()
    first.require_reselection_evidence()


def test_geometry_failure_can_reselect_without_spending_probe(probe_module,tmp_path):
    e=editor(tmp_path);e.nodes['seed_000']['result']['reference_screen_passed']=False
    current=probe_module.PDMProbe(e)
    current.require_reselection_evidence()
    assert current.context()['calls']==0


def test_no_defense_witness_is_advice_not_a_new_submission_gate(probe_module,monkeypatch,tmp_path):
    e=editor(tmp_path);e.constraints['pdm_defense_feedback']=True
    manager=probe_module.PDMProbe(e);prepare(probe_module,monkeypatch,manager,[],safe=False)
    monkeypatch.setattr(manager,'_defense_feedback',lambda *args:dict(status='completed',qualified_count=0,
        diagnostic_only=True,attack_acceptance_affected=False,global_feasibility_proven=False))
    result=manager.probe('seed_000')
    assert result['defense_feasibility']['qualified_count']==0
    assert manager.require_challenge('seed_000')['pdm_defense_succeeded'] is False


def test_diagnostic_execution_failure_stays_unknown(probe_module,monkeypatch,tmp_path):
    e=editor(tmp_path);manager=probe_module.PDMProbe(e)
    module=types.ModuleType('defense_probe_feedback')
    def fail(*args):raise RuntimeError('constructor unavailable')
    module.collect=fail;monkeypatch.setitem(sys.modules,'defense_probe_feedback',module)
    result=manager._defense_feedback(tmp_path,tmp_path)
    assert result['status']=='unavailable' and 'qualified_count' not in result
    assert not result['attack_acceptance_affected']


def test_prior_review_can_supply_control_cache_but_not_a_current_test(probe_module,monkeypatch,tmp_path):
    source=tmp_path/'prior_pdm';source.mkdir();(source/'binding.json').write_text('{}')
    report=tmp_path/'prior.json';report.write_text(json.dumps(dict(status='completed',pdm_output=str(source))))
    e=editor(tmp_path);e.constraints['revision_feedback']=dict(evidence=dict(result_path=str(report)))
    manager=probe_module.PDMProbe(e);calls=[];prepare(probe_module,monkeypatch,manager,calls)
    assert manager.context()['calls']==0 and manager.results=={}
    with pytest.raises(ValueError,match='before submitting'):manager.require_challenge('seed_000')
    manager.probe('seed_000')
    assert calls[0]['challenge_baseline_cache_path']==str(source)


def test_probe_forwards_explicit_traffic_light_mode(probe_module,monkeypatch,tmp_path):
    e=editor(tmp_path);e.constraints['pdm_ignore_traffic_lights']=True
    manager=probe_module.PDMProbe(e);calls=[]
    prepare(probe_module,monkeypatch,manager,calls,safe=False)
    manager.probe('seed_000')
    assert calls[0]['challenge_ignore_traffic_lights'] is True


def test_feedback_works_without_any_original_pdm_trajectory(probe_module,monkeypatch,tmp_path):
    import csv
    manager=probe_module.PDMProbe(editor(tmp_path))
    folder=tmp_path/'probe';(folder/'pdm/attack').mkdir(parents=True)
    row=dict(time_s=3.,ego_x_m=1.,ego_y_m=0.,ego_yaw_rad=0.,speed_mps=2.,source='nuplan_controller')
    with (folder/'pdm/attack/trajectory.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    (folder/'pdm/attack/diagnostics.json').write_text(json.dumps([dict(target_gap_m=0.,time_s=3.,observation_frame=0)]))
    monkeypatch.setattr(manager,'_plot',lambda *args:folder/'overlay.png')
    metrics=dict(contact_samples=1,first_contact={},road_violation_samples=0,min_gap_m=0.,min_target_gap_m=0.,final_speed_mps=2.)
    result=manager._feedback(folder,manager.editor.nodes['seed_000'],dict(summary=dict(results=dict(attack=metrics))))
    assert result['trajectory_samples']['pdm_baseline']==[]
    assert result['trajectory_samples']['pdm_attack'][0]['time_s']==3.
    assert not (folder/'pdm/baseline').exists()
