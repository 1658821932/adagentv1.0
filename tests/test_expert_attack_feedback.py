"""Advisory expert evidence can revise the same task once, preserving setup/budgets."""
import json
from pathlib import Path
import sys

import pytest
CORE=Path(__file__).resolve().parents[1]/'mcp_servers/scene_editing/core/adversarial'
if str(CORE) not in sys.path:sys.path.insert(0,str(CORE))
from expert_attack_feedback import collect,revision_history
from attack_budget import AttackBudgetExhausted


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    source=tmp_path/'source';prepared=tmp_path/'prepared';frozen=tmp_path/'frozen'
    frames=[{'ego2global':[[1,0,0,i],[0,1,0,0],[0,0,1,0],[0,0,0,1]]} for i in range(4)]
    for folder in (source,prepared,frozen):save(folder/'metadata/frames.json',frames)
    save(frozen/'ego-history.json',{'prepared_scene':str(prepared)})
    input_path=tmp_path/'attack/input.json'
    config=dict(prepare_ego_history=True,vadv2_self_test=True,vadv2_probe_root=str(tmp_path/'probes'),
        acceptance_policy={'max_acceleration_mps2':6.},task_contract={'owner':'user'})
    save(input_path,dict(scene=str(source),editing_scene=str(prepared),instruction='same task',constraints=config))
    trajectory=tmp_path/'expert/candidates/edit_001/trajectory.csv';trajectory.parent.mkdir(parents=True);trajectory.write_text('exact trajectory')
    review=tmp_path/'expert/review.json';save(review,dict(status='passed',candidate_id='edit_001',
        min_target_clearance_m=.375,max_acceleration_mps2=5.88))
    monkeypatch.setenv('ADVERSARIAL_CONTINUATION_ROOT',str(tmp_path))
    monkeypatch.setenv('ADVERSARIAL_ATTACK_SESSION',str(tmp_path/'shared-budget'))
    import ego_history
    monkeypatch.setattr(ego_history,'verify',lambda path:True)
    monkeypatch.setattr(ego_history,'source_scene',lambda path:source)
    attack=dict(frozen_scene_path=str(frozen),submission_path=str(tmp_path/'attack/trajectory_editing/output.json'))
    feedback_path=tmp_path/'expert/feedback.json'
    public=collect(attack,tmp_path/'expert','edit_001',review,{'max_acceleration_mps2':6.},feedback_path)
    return dict(root=tmp_path,source=source,prepared=prepared,config=config,feedback=public,path=feedback_path,trajectory=trajectory)


def test_measured_margin_is_advisory_and_reaches_main_context(evidence):
    from agent.context import summarize
    value=evidence['feedback'];assert value['acceptance_affected'] is value['automatic_revision'] is False
    assert value['measurements']['min_body_clearance_m']==.375
    assert value['measurements']['acceleration_limit_fraction']==pytest.approx(.98)
    assert summarize('ego_trajectory_editor',{'expert_feedback':value})['expert_feedback']==value


def test_one_revision_restores_prepared_history_without_old_attack_and_reuses_probe_ledger(evidence):
    config=dict(evidence['config'],revision_feedback=evidence['feedback']['revision_feedback'])
    assert revision_history(config,evidence['source'],'same task',evidence['root']/'new')==evidence['prepared']
    assert config['vadv2_probe_root']==evidence['config']['vadv2_probe_root']
    # Switching an object inside the same05 does not consume another margin revision.
    assert revision_history(config,evidence['source'],'same task',evidence['root']/'new/object_revision_01')==evidence['prepared']
    fresh=dict(evidence['config'],revision_feedback=evidence['feedback']['revision_feedback'])
    with pytest.raises(AttackBudgetExhausted,match='revision budget'):
        revision_history(fresh,evidence['source'],'same task',evidence['root']/'other')


def test_stale_or_different_task_feedback_is_rejected_without_spending_revision(evidence):
    config=dict(evidence['config'],revision_feedback=evidence['feedback']['revision_feedback'])
    with pytest.raises(ValueError,match='original attack instruction'):
        revision_history(config,evidence['source'],'changed task',evidence['root']/'new')
    changed=dict(config,task_contract={'owner':'changed'})
    with pytest.raises(ValueError,match='constraints'):
        revision_history(changed,evidence['source'],'same task',evidence['root']/'new')
    with pytest.raises(ValueError,match='operation/target locks'):
        revision_history(config,evidence['source'],'same task',evidence['root']/'new',required_operation='add_vehicle')
    evidence['trajectory'].write_text('changed after review')
    with pytest.raises(ValueError,match='matching independently checked'):
        revision_history(config,evidence['source'],'same task',evidence['root']/'new')
    assert not (evidence['root']/'shared-budget/expert-margin-revision-budget.json').exists()
