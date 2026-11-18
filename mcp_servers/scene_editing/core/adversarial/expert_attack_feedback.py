"""Measured expert margins for an optional, bounded revision of the same attack task."""
import hashlib
import json
import os
from pathlib import Path


def _hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(attack,expert_dir,candidate,review_path,acceptance,output):
    from experiment_runtime import atomic_json
    from defense_behavior import preferences
    scene=Path(attack['frozen_scene_path']);expert_dir=Path(expert_dir)
    review=json.loads(Path(review_path).read_text())
    if review.get('status')!='passed' or review.get('candidate_id')!=candidate:
        raise ValueError('Expert feedback requires an independently passed exact candidate')
    from ego_history import verify
    history_path=scene/'ego-history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else None
    original=Path(attack['submission_path']).parent.parent/'input.json'
    original_input=json.loads(original.read_text())
    prepared=Path(history['prepared_scene'] if history is not None else original_input.get('editing_scene',original_input['scene']))
    verify(prepared)
    prepared_frames=json.loads((prepared/'metadata/frames.json').read_text())
    frozen_frames=json.loads((scene/'metadata/frames.json').read_text())
    if len(prepared_frames)!=len(frozen_frames) or any(a['ego2global']!=b['ego2global'] for a,b in zip(prepared_frames,frozen_frames)):
        raise ValueError('Attack setup ego poses differ from the shared history source')
    report=dict(schema_version='expert_attack_feedback.v1',status='measured',candidate_id=candidate,
        min_body_clearance_m=review['min_target_clearance_m'],
        peak_planar_acceleration_mps2=review['max_acceleration_mps2'],
        planar_acceleration_limit_mps2=acceptance['max_acceleration_mps2'],
        acceleration_limit_fraction=review['max_acceleration_mps2']/acceptance['max_acceleration_mps2'],
        reserve_reference_m=preferences()['base_reserve_m']+preferences()['geometry_allowance_m'],
        reserve_reference_is_acceptance_threshold=False,
        guidance='Compare the original task, body clearance and motion headroom. If the expert succeeds only with a thin reserve or near-limit motion, Main may request one same-task attack revision aiming for more expert reserve while retaining exact VAD target collision. Do not weaken checks, reset budgets, alter shared history or infer global feasibility from failed searches.',
        frozen_scene_path=str(scene.resolve()),frames_sha256=_hash(scene/'metadata/frames.json'),
        expert_trajectory_path=str((expert_dir/'candidates'/candidate/'trajectory.csv').resolve()),
        expert_trajectory_sha256=_hash(expert_dir/'candidates'/candidate/'trajectory.csv'),
        expert_review_path=str(Path(review_path).resolve()),
        original_attack_input_path=str(original.resolve()),original_instruction=original_input['instruction'],
        prepared_history_scene=str(prepared.resolve()),
        prepared_history_sha256=_hash(prepared/'metadata/frames.json'),
        scope='Independent expert reference evidence, not policy improvement or a new acceptance gate')
    atomic_json(output,report)
    evidence={key:report[key] for key in ('min_body_clearance_m','peak_planar_acceleration_mps2',
        'planar_acceleration_limit_mps2','acceleration_limit_fraction','reserve_reference_m')}
    evidence.update(feedback_path=str(Path(output).resolve()))
    return dict(status='measured',measurements=evidence,guidance=report['guidance'],
        revision_feedback=dict(stage='ego_editing',outcome='review_expert_reserve',guidance=report['guidance'],evidence=evidence),
        acceptance_affected=False,automatic_revision=False,max_same_task_margin_revisions=1)


def revision_history(config,scene,instruction,output,required_operation=None,required_track=None):
    """Verify feedback provenance and restore setup without carrying the old attack actor."""
    revision=config.get('revision_feedback') or {}
    if revision.get('stage')!='ego_editing' or revision.get('outcome')!='review_expert_reserve':return None
    path=Path((revision.get('evidence') or {}).get('feedback_path','')).resolve()
    from experiment_runtime import experiments_root,atomic_json
    allowed=Path(os.getenv('ADVERSARIAL_CONTINUATION_ROOT',str(experiments_root()))).resolve()
    if not path.is_relative_to(allowed) or not path.is_file():raise ValueError('Expert feedback must be an existing report in the continuation area')
    report=json.loads(path.read_text())
    if report.get('schema_version')!='expert_attack_feedback.v1' or report.get('status')!='measured':
        raise ValueError('Unknown expert feedback version')
    if report['original_instruction']!=instruction:raise ValueError('Expert margin revision must keep the original attack instruction')
    original=json.loads(Path(report['original_attack_input_path']).read_text())
    if original.get('instruction')!=instruction:raise ValueError('Original attack task no longer matches expert feedback')
    if original.get('required_operation')!=required_operation or original.get('required_track')!=required_track:
        raise ValueError('Expert margin revision must preserve the original explicit operation/target locks')
    from attack_configuration import same_scene_constraints
    def physics(value):return {key:item for key,item in value.items() if key not in ('revision_feedback','expert_margin_revision_count')}
    if not same_scene_constraints(physics(original['constraints']),physics(config)):
        raise ValueError('Expert margin revision must keep the original task and physics constraints')
    from ego_history import source_scene,verify
    frozen=Path(report['frozen_scene_path'])
    if source_scene(frozen)!=source_scene(scene) or _hash(frozen/'metadata/frames.json')!=report['frames_sha256']:
        raise ValueError('Expert feedback is stale or belongs to another scene')
    review=json.loads(Path(report['expert_review_path']).read_text())
    if review.get('status')!='passed' or review.get('candidate_id')!=report['candidate_id'] or _hash(report['expert_trajectory_path'])!=report['expert_trajectory_sha256']:
        raise ValueError('Expert feedback has no matching independently checked trajectory')
    if report.get('prepared_history_scene') is None:raise ValueError('Expert revision needs frozen shared-history provenance')
    prepared=Path(report['prepared_history_scene']);verify(prepared)
    if _hash(prepared/'metadata/frames.json')!=report['prepared_history_sha256'] or source_scene(prepared)!=source_scene(scene):
        raise ValueError('Shared history in expert feedback changed')
    from attack_budget import constructor_budget_path
    ledger=constructor_budget_path(output).with_name('expert-margin-revision-budget.json')
    used=json.loads(ledger.read_text()).get('calls',0) if ledger.exists() else 0
    inherited=(config.get('expert_margin_revision_count')==1 and ledger.exists()
        and json.loads(ledger.read_text()).get('feedback_path')==str(path))
    if used>=1 and not inherited:
        from attack_budget import AttackBudgetExhausted
        raise AttackBudgetExhausted('Same-task expert margin revision budget exhausted; preserve original validated pair')
    for key in ('vadv2_probe_root','pdm_probe_root'):
        if original['constraints'].get(key):
            root=Path(original['constraints'][key]).resolve()
            if not root.is_relative_to(allowed):raise ValueError('Prior policy probe budget is outside continuation area')
            config[key]=str(root)
    config['expert_margin_revision_count']=1
    if not inherited:atomic_json(ledger,dict(calls=used+1,feedback_path=str(path)))
    return prepared
