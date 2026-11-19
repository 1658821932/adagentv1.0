"""Completed expert recovery must preserve the trajectory and bypass generation."""
import json
from pathlib import Path

import pytest
from mcp_servers.scene_editing.core.adversarial.completed_expert_export import reexport


@pytest.fixture
def completed(tmp_path, monkeypatch):
    import native_scene_export
    import run_adversarial_loop
    source=tmp_path/'old';output=source/'output';output.mkdir(parents=True)
    (source/'input').mkdir()
    attack=dict(scene_id='scene',scene_token='token',run_dir=str(tmp_path/'attack'),
        frozen_scene_path=str(tmp_path/'frozen'),target_track='car',source_scene_ref={})
    args=dict(attack_result=attack,target_effect='stop safely',allow_safe_stop=True,
        require_route_progress=None,reuse_completed_from=str(source),export_only=True)
    prior=dict(stage='ego_editing',status='completed',scene_id='scene',scene_token='token',
        attack_run_dir=attack['run_dir'],frozen_scene_path=attack['frozen_scene_path'],
        target_track='car',selected_candidate_id='edit_001',checks=dict(reference_screen='passed'))
    paths=dict(expert_trajectory_path=output/'defense_agent/candidates/edit_001/trajectory.csv',
        trajectory_metadata_path=output/'metadata.json',conflict_brief_path=output/'brief.json',
        submission_path=output/'summary.json')
    for key,path in paths.items():
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original '+key)
        prior[key]=str(path)
    (output/'result.json').write_text(json.dumps(prior))
    (source/'input/stage_request.json').write_text(json.dumps({'arguments':args}))
    calls=[]
    def independent(*pos):
        calls.append(('check',pos))
        return dict(checks={'collision_free':True})
    def export(*pos,**kw):
        calls.append(('export',pos,kw))
        return dict(status='exported',report_path=str(pos[2]/'export-report.json'),
            simulation_input={'frame_count':24},checks={'time_alignment':{'max_interval_drift_s':.0005}})
    monkeypatch.setattr(run_adversarial_loop,'run_independent',independent)
    monkeypatch.setattr(native_scene_export,'export',export)
    return args,prior,calls,tmp_path/'new/output'


def test_export_reuses_exact_expert_and_only_checks_and_exports(completed):
    args,prior,calls,output=completed
    result=reexport(args,output)
    assert [c[0] for c in calls]==['check','export']
    assert result['selected_candidate_id']=='edit_001'
    assert result['native_export']['status']=='exported'
    assert result['rendered_videos']=={}
    for key in ('expert_trajectory_path','trajectory_metadata_path','submission_path','conflict_brief_path'):
        assert Path(result[key]).is_relative_to(output)
        assert Path(result[key]).read_bytes()==Path(prior[key]).read_bytes()
    assert calls[0][1][1]==Path(prior['expert_trajectory_path']).parents[2]


@pytest.mark.parametrize('field',['target_effect','allow_safe_stop','require_route_progress','target_track','frozen_scene_path'])
def test_changed_task_or_attack_never_exports_saved_expert(completed,field):
    args,prior,calls,output=completed
    if field in ('target_track','frozen_scene_path'):args['attack_result'][field]='different'
    else:args[field]='changed'
    with pytest.raises(ValueError,match='changed|mismatch'):reexport(args,output)
    assert calls==[]


def test_incomplete_expert_or_missing_source_is_not_regenerated(completed):
    args,prior,calls,output=completed
    Path(prior['expert_trajectory_path']).unlink()
    with pytest.raises(ValueError,match='missing'):reexport(args,output)
    assert calls==[]
    args.pop('reuse_completed_from')
    with pytest.raises(ValueError,match='requires'):reexport(args,output)
